import os
import torch
from typing import Dict, List, Optional, OrderedDict, Tuple, Union
import warnings
from flwr.common.typing import Metrics
from hydra import compose, initialize

import flwr as fl
from utils import get_on_fit_config, fit_weighted_average
import numpy as np
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
)
from flwr.server.client_proxy import ClientProxy

warnings.filterwarnings("ignore", category=UserWarning)

NUM_ROUNDS = 5
save_path = "./results/"

### 加上去的
with initialize(config_path="conf"):
    cfg = compose(config_name="config")

# Reset the number of number
cfg.num_rounds = NUM_ROUNDS
cfg.train.num_rounds = NUM_ROUNDS

# Create output directory
if not os.path.exists(save_path):
    os.mkdir(save_path)


class SaveModelStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        # Call aggregate_fit from base class (FedAvg) to aggregate parameters and metrics
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            server_round, results, failures
        )

        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays
            print(f"Saving round {server_round} aggregated_ndarrays...")
            np.savez(
                os.path.join(save_path, f"round-{server_round}-weights.npz"),
                *aggregated_ndarrays,
            )

        return aggregated_parameters, aggregated_metrics


# Instantiate strategy.
strategy = SaveModelStrategy(
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
