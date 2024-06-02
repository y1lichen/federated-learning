from models import get_model
from flwr.common.typing import NDArrays
from flwr.common import Parameters
import flwr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from collections import OrderedDict
import torch
import glob
import os
import numpy as np


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


def set_parameters(model, parameters: NDArrays) -> None:
    """Change the parameters of the model using the given ones."""
    peft_state_dict_keys = get_peft_model_state_dict(model).keys()
    params_dict = zip(peft_state_dict_keys, parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    set_peft_model_state_dict(model, state_dict)


def get_init_weight() -> Parameters:
    list_of_files = [fname for fname in glob.glob("./results/round-*")]
    if len(list_of_files) == 0:
        print("not pre-trained model")
        return None
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    npz = np.load(latest_round_file)
    parameters = flwr.common.ndarrays_to_parameters(npz)
    return parameters
