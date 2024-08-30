import flwr as fl
import argparse

import os
from flwr.server import History
from hydra import compose, initialize
from save_model_strategy import SaveModelStrategy
from utils.utils import (
    get_on_fit_config,
    get_init_parameters,
)

with initialize(config_path="conf"):
    cfg = compose(config_name="config")
NUM_ROUNDS = cfg.num_rounds


save_path = "./results"
if not os.path.exists(save_path):
    os.mkdir(save_path)


def main(num_clients=1, num_rounds=NUM_ROUNDS) -> History:
    strategy = SaveModelStrategy(
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=get_on_fit_config,
        initial_parameters=get_init_parameters(),
    )

    # Start Flower server
    hist = fl.server.start_server(
        server_address="[::]:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )
    return hist


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "-c", "--clients", type=int, help="minimum number of clients", default=1
    )
    args = argParser.parse_args()
    hist = main(args.clients)
