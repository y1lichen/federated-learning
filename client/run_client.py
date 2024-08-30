from argparse import ArgumentParser, ArgumentTypeError
import os
from typing import Callable, Dict, Tuple
import warnings
from flwr.common import NDArrays, Scalar
from hydra import compose, initialize

import flwr as fl

from utils.dataset import get_tokenizer_and_data_collator_and_propt_formatting
from utils.client import gen_client_fn, set_parameters
from utils.custom_fds import CustomFederatedDataset


import torch

from omegaconf import DictConfig
from trl import SFTTrainer
from transformers import TrainingArguments
from peft import get_peft_model_state_dict

from models import get_model, cosine_annealing


NUM_ROUNDS = 5


with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS


# Partition dataset and get dataloaders
# We set the number of partitions to 20 for fast processing.

(
    tokenizer,
    data_collator,
    formatting_prompts_func,
) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)


def main(dataset_path: str, idx: int):
    fds = CustomFederatedDataset(dataset_path=dataset_path, partitioners={"train": 1})
    trainset = fds.load_partition(0)
    # Create output directory
    save_path = f"./results/client_{idx}"
    if not os.path.exists(save_path):
        os.mkdir(save_path)

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
            # set index
            self.index = idx
            # instantiate model
            self.model = get_model(model_cfg)
            self.trainset = trainset

        def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
            """Return the parameters of the current net."""
            state_dict = get_peft_model_state_dict(self.model)
            return [val.cpu().numpy() for _, val in state_dict.items()]

        def fit(
            self, parameters: NDArrays, config: Dict[str, Scalar]
        ) -> Tuple[NDArrays, int, Dict]:
            """Implement distributed fit function for a given client."""
            set_parameters(self.model, parameters)

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
                # 在這調batch size
                dataset_batch_size=512,
            )

            # Do local training
            results = trainer.train()
            # save model
            torch.save(
                self.model.state_dict(),
                self.save_path + f"round_{config['current_round']}.pth",
            )
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
        "-f",
        "--file",
        dest="filename",
        required=True,
        type=validate_file,
        help="input file",
        metavar="FILE",
    )
    parser.add_argument(
        "-i",
        "--idx",
        dest="index",
        required=True,
        type=int,
        help="id of the client",
        metavar="INDEX",
    )
    args = parser.parse_args()
    main(args.filename, args.index)
    # print(args.filename)
