import sys
from typing import List, Tuple
from pathlib import Path

import flwr as fl
from flwr.common import Metrics

from utils import Parameters

params = Parameters()
config_file = sys.argv[1]
params.GetParams(config_file)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

def equal_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [ m["accuracy"] for num_examples, m in metrics]
    return {"accuracy": sum(accuracies) }

if params.metrics_aggregation == "weighted_average":
    metrics = weighted_average
elif params.metrics_aggregation == "equal_average":
    metrics = equal_average

if params.strategy == "FedAvg":
    print("================================")
    print(params.min_fit_clients, params.min_evaluate_clients,  params.min_available_clients)
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=metrics,
    min_fit_clients = params.min_fit_clients,
    min_evaluate_clients = params.min_evaluate_clients,
    min_available_clients = params.min_available_clients)

elif params.strategy == "FedOps":
    strategy = fl.server.strategy.FedOpt(evaluate_metrics_aggregation_fn=metrics,
    min_fit_clients = params.min_fit_clients,
    min_evaluate_clients = params.min_evaluate_clients,
    min_available_clients = params.min_available_clients)
elif params.strategy == "FedProx":
    strategy = fl.server.strategy.FedProx(evaluate_metrics_aggregation_fn=metrics,
    min_fit_clients = params.min_fit_clients,
    min_evaluate_clients = params.min_evaluate_clients,
    min_available_clients = params.min_available_clients)

if params.use_certificates == True or params.use_certificates == "True":
    fl.server.start_server(
        server_address=params.server_address,
        #server_address="[SERVER_IP]:[PORT]"
            certificates=(
                Path("./src/certificates/rootCA_cert.pem").read_bytes(),
                Path("./src/certificates/server_cert.pem").read_bytes(),
                Path("./src/certificates/server_key.pem").read_bytes(),
            ),    config=fl.server.ServerConfig(num_rounds=params.num_rounds),
        strategy=strategy,
    )
else:
    fl.server.start_server(
    server_address=params.server_address,
    #server_address="[SERVER_IP]:[PORT]"
    #    certificates=(
    #        Path("./src/certificates/rootCA_cert.pem").read_bytes(),
    #        Path("./src/certificates/server_cert.pem").read_bytes(),
    #        Path("./src/certificates/server_key.pem").read_bytes(),
    #    ),    config=fl.server.ServerConfig(num_rounds=params.num_rounds),
    strategy=strategy,
)
