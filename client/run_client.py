from argparse import ArgumentParser, ArgumentTypeError
import os
from typing import Callable, Dict, Tuple
import warnings
from flwr.common import NDArrays
from hydra import compose, initialize

import flwr as fl
from flwr_datasets import FederatedDataset

from utils.dataset import get_tokenizer_and_data_collator_and_propt_formatting
from utils.client import gen_client_fn
from utils.utils import get_on_fit_config, fit_weighted_average
from utils.custom_fds import CustomFederatedDataset

from argparse import ArgumentParser
from collections import OrderedDict

import torch
from flwr.common.typing import NDArrays, Scalar
from omegaconf import DictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import get_peft_model_state_dict, set_peft_model_state_dict

from models import get_model, cosine_annealing


NUM_ROUNDS = 5
save_path = "./results/"

with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Partition dataset and get dataloaders
# We set the number of partitions to 20 for fast processing.

(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)


def main(dataset_path: str):
    fds = CustomFederatedDataset(
        dataset_path=dataset_path, partitioners={"train": cfg.num_clients}
    )
    trainset = fds.load_partition(0)

    # Flower client
    class CustomClient(fl.client.NumPyClient):
        def __init__(
            self,
            model_cfg: DictConfig,
            train_cfg: DictConfig,
            trainset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            save_path,
        ):  # pylint: disable=too-many-arguments
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.train_cfg = train_cfg
            self.training_argumnets = TrainingArguments(**train_cfg.training_arguments)
            self.tokenizer = tokenizer
            self.formatting_prompts_func = formatting_prompts_func
            self.data_collator = data_collator
            self.save_path = save_path

            # instantiate model
            self.model = get_model(model_cfg)

            self.trainset = trainset

        def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
            """Return the parameters of the current net."""

            state_dict = get_peft_model_state_dict(self.model)
            return [val.cpu().numpy() for _, val in state_dict.items()]

        def set_parameters(self, model, parameters: NDArrays) -> None:
            """Change the parameters of the model using the given ones."""
            peft_state_dict_keys = get_peft_model_state_dict(model).keys()
            params_dict = zip(peft_state_dict_keys, parameters)
            state_dict = OrderedDict(
                {k: torch.tensor(v, requires_grad=True) for k, v in params_dict}
            )
            set_peft_model_state_dict(model, state_dict)

        def fit(
            self, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Tuple[NDArrays, int, Dict]:
            """Implement distributed fit function for a given client."""
            self.set_parameters(self.model, parameters)

            new_lr = cosine_annealing(
                int(config["current_round"]),
                self.train_cfg.num_rounds,
                self.train_cfg.learning_rate_max,
                self.train_cfg.learning_rate_min,
            )

            self.training_argumnets.learning_rate = new_lr
            self.training_argumnets.output_dir = self.save_path

            # Construct trainer
            trainer = SFTTrainer(
                model=self.model,
                tokenizer=self.tokenizer,
                args=self.training_argumnets,
                max_seq_length=self.train_cfg.seq_length,
                train_dataset=self.trainset,
                formatting_func=self.formatting_prompts_func,
                data_collator=self.data_collator,
            )

            # Do local training
            results = trainer.train()

            return (
                self.get_parameters({}),
                len(self.trainset),
                {"train_loss": results.training_loss},
            )

    # Start client
    fl.client.start_client(
        server_address="127.0.0.1:8080",
        client=CustomClient(
            cfg.model,
            cfg.train,
            trainset,
            tokenizer,
            formatting_prompts_func,
            data_collator,
            save_path,
        ).to_client(),
    )


# 確定有file
def validate_file(f):
    if not os.path.exists(f):
        # Argparse uses the ArgumentTypeError to give a rejection message like:
        # error: argument input: x does not exist
        raise ArgumentTypeError("{0} does not exist".format(f))
    return f


if __name__ == "__main__":
    parser = ArgumentParser("GetFilePath")
    parser.add_argument(
        "-i",
        "--input",
        dest="filename",
        required=True,
        type=validate_file,
        help="input file",
        metavar="FILE",
    )
    args = parser.parse_args()
    fds = CustomFederatedDataset(
        dataset_path=args.filename, partitioners={"train": cfg.num_clients}
    )
    main(args.filename)
    # print(args.filename)
