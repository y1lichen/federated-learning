import flwr as fl
import pickle


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
            # np.savez(
            #     os.path.join(save_path, f"round-{server_round}-weights.npz"),
            #     *aggregated_ndarrays,
            # )
            # data = np.load(os.path.join(save_path, f"round-{server_round}-weights.npz"))
            # arr = data["arr"].item()
            with open(os.path.join(save_path, f"round-{server_round}-weights.pkl"), "wb") as f:
                pickle.dump(aggregated_ndarrays, f)

        return aggregated_parameters, aggregated_metrics
