import os
import sys
import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
from collections import OrderedDict

import flwr as fl
from flwr.common import Metrics

from utils import Parameters
from Models.MLP.model import MLP_SODEN
from model_wrapper import ModelWrapper

# Add 'lib' directory to Python's module search path
sys.path.append("/home/jorge/work_dir/nouman/passport/python-auto-metadata-collection/lib")
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))

from ai4hf_passport_torch import TorchMetadataCollectionAPI
from ai4hf_passport_models import LearningStage, EvaluationMeasure, Model, LearningStageType, EvaluationMeasureType

params = Parameters()
config_file = sys.argv[1]
params.GetParams(config_file)

#def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
##AQUI
#    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
#    examples = [num_examples for num_examples, _ in metrics]
#    return {"accuracy": sum(accuracies) / sum(examples)}

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    aggregated = {}
    total_examples = sum(num_examples for num_examples, _ in metrics)

    if total_examples == 0:
        return {}

    # Iterar sobre todas las claves presentes en los diccionarios de métricas
    keys = metrics[0][1].keys() if metrics else []
    
    for key in keys:
        weighted_sum = sum(num_examples * m[key] for num_examples, m in metrics)
        aggregated[key] = weighted_sum / total_examples
    #print("WEIGHTED AVERAGE :: aggregated", aggregated)
    with open("metrics.json", "w") as f:
        json.dump(aggregated, f, indent=4) 
    # ********** * * * * *  *  *  *  *    *   *  *  * * * * *************
    return aggregated

def equal_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [ m["accuracy"] for num_examples, m in metrics]
    return {"accuracy": sum(accuracies) }

if params.metrics_aggregation == "weighted_average":
    metrics = weighted_average
elif params.metrics_aggregation == "equal_average":
    metrics = equal_average

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        final_model = fl.common.parameters_to_ndarrays(aggregated_parameters)
        np.savez("final_model.npz", **{f"arr_{i}": arr for i, arr in enumerate(final_model)})
        #np.savez("final_model.npz", final_model)
        print("SERVER::CustomStrategy::aggregate_fit::final_model",type(final_model),type(final_model[0])) #,final_model)
        print(f"Storaged model after round {rnd}.")
        return aggregated_parameters, _

if params.strategy == "FedAvg":
    """
    strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=metrics,
    min_fit_clients = params.min_fit_clients,
    min_evaluate_clients = params.min_evaluate_clients,
    min_available_clients = params.min_available_clients)
    """
    strategy = CustomStrategy(evaluate_metrics_aggregation_fn=metrics,
                              min_fit_clients=params.min_fit_clients,
                              min_evaluate_clients=params.min_evaluate_clients,
                              min_available_clients=params.min_available_clients)
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
    #    ),    
    config=fl.server.ServerConfig(num_rounds=params.num_rounds),
    strategy=strategy,
)
# ******** * * *  *  *   *    *   *   *   *  * * * * * * **************
if params.local_model == "MLP":
    configurations = [
    {
    'features': 'maggic', 'feature_size': 13, 'mlp_hidden_sizes': [16, 32, 16], 'mlp_output_size': 16, 
    'ode_hidden_size': 16, 'ode_num_layers': 2, 'ode_batch_norm': False,'time_nums': 62
    },
    {
    'features': 'maggic_plus', 'feature_size': 19, 'mlp_hidden_sizes': [32, 64, 32], 'mlp_output_size': 16, 
    'ode_hidden_size': 16, 'ode_num_layers': 2, 'ode_batch_norm': False,'time_nums': 62
    }
    ]
    print("LEYO PARAMETROS")
    if params.features == "maggic":
        config_selector = 0
    elif params.features == "maggic_plus":
        config_selector = 1
    print("CONFIG SELECTOR", config_selector)
    config = configurations[config_selector]
    #Aqui primer problema: esto no debería estar alambrado
    #self.model_folder = '/home/jorge/work_dir/nouman/AI4HF-OXF-Modelling/new_models'
    # o sería mejor ponerlo como variable
    model_folder = params.log_path #'data'
    os.makedirs(model_folder, exist_ok=True)  # Crea el directorio si no existe
    model_file = os.path.join(model_folder, f"{config['features']}_model.pth")
    model = MLP_SODEN(config)
    model.suffix = config['features']
    model_keys = model.state_dict().keys()
    parameters_ = np.load("final_model.npz")
    parameters = [parameters_[key] for key in parameters_.files]

    print("server::model_keys",model_keys,type(model_keys))
    print("server::parameters",parameters,type(parameters))
# ==================================================================================
#    torch.save(model.state_dict(), f'{model_folder}/{model.suffix}_model.pth')
# ==================================================================================
    #model = model.to(device)
else:
    model = ModelWrapper(params)
    #self.model = model.to(device)

#def set_parameters(self, parameters:List[np.ndarray]):
if params.local_model == "MLP":
    # # fFalta sacar el model_keys de algun lado           
    params_dict = zip(model_keys, parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    # tenemos algo asi? quizas lo mas facil sea escribir un archivo pth y 
    # cargar el archivo com model load

    print("SET PARAMETERS::LOAD STATE DICT")
    model.load_state_dict(state_dict, strict=True)
else:
    model.train()
    # Si esto del self.model.train no funciona porque no reconoce la
    # función entonces deberías sustituírla por nuestra train:
    # train(self.model,params)
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# ******** * * *  *  *   *    *   *   *   *  * * * * * * **************
# Construct an api client for interacting with AI4HF passport server

api_client = TorchMetadataCollectionAPI(
        passport_server_url="http://localhost:80/ai4hf/passport/api",
        study_id="1",
        organization_id="1",
        username="data_scientist",
        password="data_scientist"
    )

# Provide learning stages
learning_stages = [
    LearningStage(learningStageType = LearningStageType.TRAINING,
                  datasetPercentage = str(params.train_size)),
    LearningStage(learningStageType = LearningStageType.TEST,
                  datasetPercentage = str(params.test_size)),
    LearningStage(learningStageType = LearningStageType.VALIDATION,
                  datasetPercentage = str(params.val_size))
]

with open("metrics.json", "r") as f:
    metrics_file = json.load(f)
# {'tempauroc': 0.5018877581090273, 'tempprc': 0.5009819110168369, 'loss': 2.470166275946147}

# Provide evaluation measures
evaluation_measures = [
    EvaluationMeasure(EvaluationMeasureType.MAE,
                      value = str(metrics_file["loss"])),
    EvaluationMeasure(EvaluationMeasureType.ROC,
                      value = str(metrics_file["tempprc"])),
    EvaluationMeasure(EvaluationMeasureType.AUC,
                      value = str(metrics_file["tempauroc"]))
]
"""
        loss defined :
        loss_fct = SurvODELoss(reduction='mean')
"""
# Provide model details
model_info = Model(
    name = "test",
    modelType = "MLP",
    version = "1.0")

# Call this function with your model object
api_client.submit_results_to_ai4hf_passport(model, learning_stages, evaluation_measures, model_info)

## COSAS POR AHCER:
# Hacer variable el directorio del lib passport
# personalizar el lerning stage y el evaluation measures
