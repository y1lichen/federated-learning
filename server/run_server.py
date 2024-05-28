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
# 主要功能都會在strategy中處理
# 因為想要每個client有一樣權重，所以用FedAvg。見以下詳細資料
# https://biic.ee.nthu.edu.tw/blog-detail.php?id=2
# https://www.royc30ne.com/fedavg/
strategy = fl.server.strategy.FedAvg(
    # 最少要兩人個client上傳model權重才會執行FL。如果只有一個model上傳權重，server會先擱著取得的權重，直到有第二個model權重
    # 不可直接把min_available_client改成1，在fit時會噴error，別直接這麼做。見single-client-server的實作
    min_available_clients=2,
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # no client evaluation
    on_fit_config_fn=get_on_fit_config(),
    fit_metrics_aggregation_fn=fit_weighted_average,
    evaluate_fn=get_evaluate_fn(
        cfg, cfg.train.save_every_round, cfg.num_rounds, save_path
    ),
)

server = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
)
