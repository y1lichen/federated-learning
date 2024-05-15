import os
from typing import List, Tuple
import warnings
from flwr.common.typing import Metrics
from hydra import compose, initialize

import flwr as fl
from flwr_datasets import FederatedDataset

from dataset import get_tokenizer_and_data_collator_and_propt_formatting
from client import gen_client_fn
from utils import get_on_fit_config, fit_weighted_average


warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 5
save_path = "./results/"


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    examples = [num_examples for num_examples, _ in metrics]

    # Multiply accuracy of each client by number of examples used
    train_losses = [num_examples * m["train_loss"] for num_examples, m in metrics]
    train_accuracies = [
        num_examples * m["train_accuracy"] for num_examples, m in metrics
    ]
    val_losses = [num_examples * m["val_loss"] for num_examples, m in metrics]
    val_accuracies = [num_examples * m["val_accuracy"] for num_examples, m in metrics]

    # Aggregate and return custom metric (weighted average)
    return {
        "train_loss": (train_losses) / sum(examples),
        "train_accuracy": sum(train_accuracies) / sum(examples),
        "val_loss": sum(val_losses) / sum(examples),
        "val_accuracy": sum(val_accuracies) / sum(examples),
    }


# Instantiate strategy.
strategy = fl.server.strategy.FedAvg(
    min_available_clients=2,  # Simulate a 2-client setting
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # no client evaluation
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
)

# ServerApp for Flower-Next
server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
