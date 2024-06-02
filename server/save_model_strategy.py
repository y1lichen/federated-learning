import flwr as fl
import numpy as np


class SaveModelStrategy(fl.server.strategy.FedAvg):

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
            np.savez(f"./results/round-{rnd}-weights.npz", *aggregated_ndarrays)

        return aggregated_parameters, aggregated_metrics
