from flwr.common.typing import NDArrays
from flwr.common import Parameters
import flwr
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from collections import OrderedDict
import torch
import glob
import os
import numpy as np
from typing import List
from models import get_model


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


def get_init_parameters_as_statedict(keys: List[str]):
    list_of_files = [fname for fname in glob.glob("./results/round-*")]
    if len(list_of_files) == 0:
        print("not pre-trained model")
        return None
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    data = np.load(latest_round_file)
    # Ensure that all items in the data are numerical and convert to PyTorch tensors
    state_dict = {
        key: torch.tensor(data[f"arr_{i}"])
        for i, key in enumerate(keys)
        for key, v in data.items()
        if isinstance(v, np.ndarray)
        and v.dtype
        in [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
            np.bool_,
        ]
    }
    return state_dict


def get_init_parameters() -> Parameters:
    list_of_files = [fname for fname in glob.glob("./results/round-*")]
    if len(list_of_files) == 0:
        print("not pre-trained model")
        return None
    latest_round_file = max(list_of_files, key=os.path.getctime)
    print("Loading pre-trained model from: ", latest_round_file)
    data = np.load(latest_round_file)
    # Ensure that all items in the data are numerical and convert to PyTorch tensors
    state_dict = {
        k: torch.tensor(v)
        for k, v in data.items()
        if isinstance(v, np.ndarray)
        and v.dtype
        in [
            np.float64,
            np.float32,
            np.float16,
            np.int64,
            np.int32,
            np.int16,
            np.int8,
            np.uint64,
            np.uint32,
            np.uint16,
            np.uint8,
            np.bool_,
        ]
    }

    # Convert the state_dict to a list of NumPy arrays for Flower
    initial_parameters = [v.numpy() for v in state_dict.values()]
    initial_parameters = flwr.common.ndarrays_to_parameters(initial_parameters)
    return initial_parameters


def get_evaluate_fn(cfg, save_every_round, total_round, save_path):

    def evaluate(server_round: int, parameters, config):
        # Save model
        if server_round != 0 and (
            server_round == total_round or server_round % save_every_round == 0
        ):
            # Init model
            model = get_model(cfg.model)
            set_parameters(model, parameters)
            model.save_pretrained(f"results/peft_{server_round}")

        return 0.0, {}

    return evaluate
