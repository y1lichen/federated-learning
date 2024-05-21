import os
from typing import List, Tuple
import warnings
from flwr.common.typing import Metrics
from hydra import compose, initialize

import flwr as fl
from flwr_datasets import FederatedDataset

from utils.dataset import get_tokenizer_and_data_collator_and_propt_formatting
from utils.utils import get_on_fit_config, fit_weighted_average, get_evaluate_fn

warnings.filterwarnings("ignore", category=UserWarning)

with initialize(config_path="conf"):
    cfg = compose(config_name="config")
NUM_ROUNDS = cfg.num_rounds

save_path = "./results/"
if not os.path.exists(save_path):
    os.mkdir(save_path)

# Instantiate strategy.
strategy = fl.server.strategy.FedAvg(
    min_available_clients=1, # 只要一個client就可以進行finetune了
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # no client evaluation
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
    evaluate_fn=get_evaluate_fn(cfg.model, cfg.train.save_every_round, cfg.num_rounds, save_path)
)

# ServerApp for Flower-Next
server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
