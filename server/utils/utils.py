from models import get_model
from flwr.common.typing import NDArrays
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from collections import OrderedDict
import torch


# Get a function that will be used to construct the config that the client's
# fit() method will receive
def get_on_fit_config():
    def fit_config_fn(server_round: int):
        fit_config = {"current_round": server_round}
        return fit_config

    return fit_config_fn


def fit_weighted_average(metrics):
    """Aggregation function for (federated) evaluation metrics."""
    # Multiply accuracy of each client by number of examples used
    losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"train_loss": sum(losses) / sum(examples)}


def get_evaluate_fn(cfg, save_every_round, total_round, save_path):
    """Return an evaluation function for saving global model."""
    model_cfg = cfg.model

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(model_cfg)
            set_parameters(model, parameters)

            model.save_pretrained(f"{save_path}/peft_{server_round}")

        return 0.0, {}

    return evaluate


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)
