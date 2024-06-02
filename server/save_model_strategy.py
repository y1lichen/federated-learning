import flwr
from utils.utils import get_on_fit_config, fit_weighted_average


class SaveModelStrategy(flwr.server.strategy.FedAvg):

    def fit_metrics_aggregation_fn(self):
        return fit_weighted_average

    def get_on_fit_config(self):
        return get_on_fit_config

    def aggregate_fit(
        self,
        rnd: int,
        results,
        failures,
    ):
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(
            rnd, results, failures
        )
        if aggregated_parameters is not None:
            # Convert `Parameters` to `List[np.ndarray]`
            aggregated_ndarrays: List[np.ndarray] = fl.common.parameters_to_ndarrays(
                aggregated_parameters
            )

            # Save aggregated_ndarrays
            print(f"Saving round {rnd} aggregated_ndarrays...")
            np.savez(f"round-{rnd}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
