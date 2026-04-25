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

from Models.MLP.model import MLP_SODEN
from model_wrapper import ModelWrapper

# Add 'lib' directory to Python's module search path
sys.path.append("/home/jorge/work_dir/nouman/passport/python-auto-metadata-collection/lib")
#sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "lib")))

from ai4hf_passport_torch import TorchMetadataCollectionAPI
from ai4hf_passport_models import LearningStage, EvaluationMeasure, Model, LearningStageType, EvaluationMeasureType

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

class CustomStrategy(fl.server.strategy.FedAvg):
    def aggregate_fit(self, rnd, results, failures):
        aggregated_parameters, _ = super().aggregate_fit(rnd, results, failures)
        final_model = fl.common.parameters_to_ndarrays(aggregated_parameters)
        np.savez("final_model.npz", **{f"arr_{i}": arr for i, arr in enumerate(final_model)})
        #np.savez("final_model.npz", final_model)
        print("SERVER::CustomStrategy::aggregate_fit::final_model",type(final_model),type(final_model[0])) #,final_model)
        print(f"Storaged model after round {rnd}.")
        return aggregated_parameters, _

## COSAS POR AHCER:
# Hacer variable el directorio del lib passport
# personalizar el lerning stage y el evaluation measures
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Reads parameters from command line.")
    # General settings
    parser.add_argument("--model", type=str, default=None, help="Model to train")
    parser.add_argument("--task", type=str, default=None, help="Task to train")
    parser.add_argument("--num_rounds", type=int, default=50, help="Number of federated iterations")
    parser.add_argument("--num_clients", type=int, default=1, help="Number of clients")
    parser.add_argument("--min_fit_clients", type=int, default=0, help="Minimum number of fit clients")
    parser.add_argument("--min_evaluate_clients", type=int, default=0, help="Minimum number of evaluate clients")
    parser.add_argument("--min_available_clients", type=int, default=0, help="Minimum number of available clients")
    parser.add_argument("--seed", type=int, default=42, help="Seed")

    # Strategy settings
    parser.add_argument("--strategy", type=str, default="FedAvg",  help="Metrics")
    parser.add_argument("--smooth_method", type=str, default="EqualVoting", help="Weight smoothing")
    parser.add_argument("--smoothing_strenght", type=float, default=0.5, help="Smoothing strenght")
    parser.add_argument("--dropout_method", type=str, default=None, help="Determines if dropout is used")
    parser.add_argument("--dropout_percentage", type=float, default=0.0, help="Ratio of dropout nodes")
    parser.add_argument("--checkpoint_selection_metric", type=str, default="precision", help="Metric used for checkpoints")
    parser.add_argument("--metrics_aggregation", type=str, default="weighted_average",  help="Metrics")
    parser.add_argument("--experiment_name", type=str, default="experiment_1", help="Experiment directory")

    # Model specific RandomForest settings
    parser.add_argument("--balanced", type=str, default=None, help="Random forest balanced")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    parser.add_argument("--max_depth", type=int, default=2, help="Max depth")
    parser.add_argument("--class_weight", type=str, default="balanced", help="Class weight")
    parser.add_argument("--levelOfDetail", type=str, default="DecisionTree", help="Level of detail")
    parser.add_argument("--regression_criterion", type=str, default="squared_error", help="Criterion for training")

    # Model specifc XGB settings
    parser.add_argument("--booster", type=str, default="gbtree", help="Booster to use: gbtree, gblinear or dart")
    parser.add_argument("--tree_method", type=str, default="hist", help="Tree method: exact, approx hist")
    parser.add_argument("--train_method", type=str, default="bagging", help="Train method: bagging, cyclic")
    parser.add_argument("--eta", type=float, default=0.1, help="ETA value")

    # Model specifc Cox settings
    parser.add_argument("--l1_penalty", type=float, default=0.0, help="L1 Penalty")

    # *******************************************************************************************
    # Parámetros que deberían eliminarse
    parser.add_argument("--sandbox_path", type=str, default="/sandbox", help="Sandbox path to use")
    parser.add_argument("--local_port", type=int, default=8081, help="Local port")
    parser.add_argument("--production_mode", type=str, default="True",  help="Production mode")
    #parser.add_argument("--certs_path", type=str, default="./", help="Certificates path")
    parser.add_argument("--n_features", type=int, default=0, help="Number of features")
    parser.add_argument("--n_feats", type=int, default=0, help="Number of features")
    parser.add_argument("--n_out", type=int, default=0, help="Number of outputs")
    # *******************************************************************************************

    args = parser.parse_args()
    config = vars(args)
    config = CheckServerConfig(config)

    if config['metrics_aggregation'] == "weighted_average":
        metrics = weighted_average
    elif config['metrics_aggregation'] == "equal_average":
        metrics = equal_average

    if config["production_mode"] == "True":
#        data_path = os.getenv("DATA_PATH")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        sandbox_path = os.getenv("SANDBOX_PATH")

        certificates = (
            Path(os.getenv("FLOWER_SSL_CACERT")).read_bytes(),
            Path('certificates/server.pem').read_bytes(),
            Path('certificates/server.key').read_bytes(),
        )
    else:
#        data_path = config["data_path"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        sandbox_path = config["sandbox_path"]
        certificates = None

    # Create experiment directory
    experiment_dir = Path(os.path.join(sandbox_path,config["experiment_name"]))
    experiment_dir.mkdir(parents=True, exist_ok=True)
    sandbox_log_file = Path(os.path.join(experiment_dir, "log_server.txt"))
    config["experiment_dir"] = experiment_dir

    # Checkpoint directory for saving the model
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)


    if config['strategy'] == "FedAvg":
        """
        strategy = fl.server.strategy.FedAvg(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config['min_fit_clients'],
        min_evaluate_clients = config['min_evaluate_clients'],
        min_available_clients = config['min_available_clients'])
        """
        strategy = CustomStrategy(evaluate_metrics_aggregation_fn=metrics,
                                min_fit_clients=config['min_fit_clients'],
                                min_evaluate_clients=config['min_evaluate_clients'],
                                min_available_clients=config['min_available_clients'])
    elif config['strategy'] == "FedOps":
        strategy = fl.server.strategy.FedOpt(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config['min_fit_clients'],
        min_evaluate_clients = config['min_evaluate_clients'],
        min_available_clients = config['min_available_clients'])
    elif config['strategy'] == "FedProx":
        strategy = fl.server.strategy.FedProx(evaluate_metrics_aggregation_fn=metrics,
        min_fit_clients = config['min_fit_clients'],
        min_evaluate_clients = config['min_evaluate_clients'],
        min_available_clients = config['min_available_clients'])
    if config['use_certificates'] == True or config['use_certificates'] == "True":
        fl.server.start_server(
            server_address=config['server_address'],
            #server_address="[SERVER_IP]:[PORT]"
                certificates=(
                    Path("./src/certificates/rootCA_cert.pem").read_bytes(),
                    Path("./src/certificates/server_cert.pem").read_bytes(),
                    Path("./src/certificates/server_key.pem").read_bytes(),
                ),    config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
            strategy=strategy,
        )
    else:
        fl.server.start_server(
        server_address=config['server_address'],
        #server_address="[SERVER_IP]:[PORT]"
        #    certificates=(
        #        Path("./src/certificates/rootCA_cert.pem").read_bytes(),
        #        Path("./src/certificates/server_cert.pem").read_bytes(),
        #        Path("./src/certificates/server_key.pem").read_bytes(),
        #    ),    
        config=fl.server.ServerConfig(num_rounds=config['num_rounds']),
        strategy=strategy,
    )
    # ******** * * *  *  *   *    *   *   *   *  * * * * * * **************
    if config['model'] == "MLP":
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
        if config['features'] == "maggic":
            config_selector = 0
        elif config['features'] == "maggic_plus":
            config_selector = 1
        print("CONFIG SELECTOR", config_selector)
        config = configurations[config_selector]
        #Aqui primer problema: esto no debería estar alambrado
        #self.model_folder = '/home/jorge/work_dir/nouman/AI4HF-OXF-Modelling/new_models'
        # o sería mejor ponerlo como variable
        model_folder = config['log_path'] #'data'
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
        model = ModelWrapper(config)
        #self.model = model.to(device)

    #def set_parameters(self, parameters:List[np.ndarray]):
    if config['local_model'] == "MLP":
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
                    datasetPercentage = int(100*config['train_size'])),
        LearningStage(learningStageType = LearningStageType.TEST,
                    datasetPercentage = int(100*config['test_size'])),
        LearningStage(learningStageType = LearningStageType.VALIDATION,
                    datasetPercentage = int(100*config['val_size']))
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