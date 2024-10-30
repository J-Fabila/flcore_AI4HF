import os
import sys
import torch
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple

from pathlib import Path

from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from train import train
from test import test
from utils import Parameters

from mlp_utils import load_json_config
from data_mlp import process_imputed_data
from Models.MLP.model import MLP_SODEN
from mlp_trainer import main_training_loop

from model_wrapper import ModelWrapper
from dataloaders import MMsDataSet,LightningWrapperData, MLPWrapperData

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, params):
        self.params = params
        print(" FLOWER CLIENT INIT")
        if torch.cuda.is_available() and params.device == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")
        self.device = device
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

            if params.features == "maggic":
                config_selector = 0
            elif params.features == "maggic_plus":
                config_selector = 1

            self.config = configurations[config_selector]

            self.model_folder = '/home/jorge/work_dir/nouman/AI4HF-OXF-Modelling/new_models'
            self.model_file = os.path.join(self.model_folder, f"{self.config['features']}_model.pth")

            model = MLP_SODEN(self.config)
            self.model = model.to(device)

            model.suffix = self.config['features']
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=self.params.lr_patience, verbose=True)

        else:
            model = ModelWrapper(params)
            self.model = model.to(device)

        if params.local_model == "MLP":
            if params.MLP_preprocess:
            #************* * * *  *  *  *   *  Data preprocessing  *    *  *  *  *  * * *************
                print(" CORRIENDO EL PREPROCESSING")
                configuration = load_json_config(params.configuration_file)
                # Process imputed data
                process_imputed_data(configuration)
                print("TERMINANDO PREPROCESING")
            #************* * * *  *  *  *   *   *      *     *     *    *  *  *  *  * * *************
            else:
                pass

        else:
            if params.dataset == "MMs":
                self.dataset = MMsDataSet(params)
                
                self.dataset = MLPWrapperData(params)

            elif params.dataset == "LightningWrapperData":
                self.dataset = LightningWrapperData(params)
            else:
                print("Dataset not available")
                exit()
            print("CLIENT::FLOWER CLIENT")    
            self.dataset.setup("fit")

    def get_parameters(self, config): # config not needed at all
        print(f"[Client {self.params.client_id}] get_parameters")
        if self.params.local_model == "MLP":
            loc_model = torch.load(self.model_file)
            return [val.cpu().numpy() for val in loc_model.values()]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters:List[np.ndarray]):
        print(f"[Client {self.params.client_id}] set_parameters")
        if self.params.local_model == "MLP":
            # # fFalta sacar el model_keys de algun lado           
            params_dict = zip(self.model_keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # tenemos algo asi? quizas lo mas facil sea escribir un archivo pth y 
            # cargar el archivo com model load
            self.model.load_state_dict(state_dict, strict=True)
        else:
            self.model.train()
            # Si esto del self.model.train no funciona porque no reconoce la
            # función entonces deberías sustituírla por nuestra train:
            # train(self.model,params)
            params_dict = zip(self.model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, params):
        if self.params.local_model == "MLP":
            print("PUNTO DE REVISION ")
            train_filepath = os.path.join(self.params.data_folder, f"train_{self.config['features']}.pt")
            test_filepath = os.path.join(self.params.data_folder, f"valid_{self.config['features']}.pt")
            model_path = self.model_folder # el directorio de log de los params

            results = main_training_loop(self.model, train_filepath, test_filepath,
                                    model_path, self.optimizer, self.params.epochs,
                                   self.params.lr_patience, self.scheduler, self.device)
            trainloader_dataset_len = 1.0 #self.dataset.train_size                       
            return self.get_parameters(config={}), trainloader_dataset_len, {}
        else:
            print(" ***************************************** FIT self.params.client_id ", self.params)
            print(f"[Client {self.params.client_id}] fit")
            self.set_parameters(parameters)
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
            train(self.model,self.params,self.dataset)
    ######## Aquí el train ya incluye su propio wandb. Pero podríamos loggear
    #        las losses o algo asi del proceso de entrenamiento
            trainloader_dataset_len = self.dataset.train_size
            return self.get_parameters(config={}), trainloader_dataset_len, {}
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
    """
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)

        trainer = pl.Trainer()
        results = trainer.test(self.model, self.test_loader)
        loss = results[0]["test_loss"]

        return loss, 10000, {"loss": loss}
    """
    def evaluate(self, parameters, params):
        if self.params.local_model == "MLP":
            pass
        else:
            # parameters es una lista y params un diccionario vacio
            # En principio aqui aceptamos params, pero no depende de nosotros pasar params,
            # flower pasa los parametros que le salen de los huevos
            print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[Client {self.params.client_id}] evaluate")
            self.set_parameters(parameters)
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
            loss, accuracy = test(self.model, self.dataset)
            # Aquí si que podríamos loggear pero sería para cada cliente individual
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
    #        return float(loss), len(testloader_dataset_len), {"accuracy": float(accuracy)}
    #        return float(loss), 100, {"accuracy": float(accuracy)}
            #print("************* * * *  *  *  *   *    *    *  *  *  *  * * *************")
            #print("loss acc", loss, accuracy)
            #print("************* * * *  *  *  *   *    *    *  *  *  *  * * *************")
            return float(loss), self.dataset.test_size, {"accuracy": float(accuracy)}
    #        return float(loss), len(testloader_dataset_len), {"accuracy": float(accuracy), "loss": float(loss)}

def main():
    ## OJO: aqui falta cambiar el len(dataset) en evaluate y fit
    print(" MAIN DEL CLIENTE =======================================")
    print(" client sim  :: main :: inicia")
    if len(sys.argv) > 1:
        params = Parameters()
        params.GetParams(sys.argv[1])
    else:
        print("*.yaml input file needed as command line argument")
        exit()

    # No needed to load data since that will be done by training torch lightning wrapper

    # Creation of the model instances
    print(" ################################################### INITIAL PARAMS", params)

    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
    # Aquí empieza lo interesante. Lo demás está en principio listo.
    # Ten cuidado de generar los clientes con cuidado, sobre todo los
    # get parameters y set parameters y que sea consistente con la integración pl
    client = FlowerClient(params).to_client()
    print(" ________________________________________________  Cliente generado")
    if params.use_certificates == True or params.use_certificates == "True":
        fl.client.start_client(
             server_address=params.server_address,
             root_certificates=Path("./src/certificates/rootCA_cert.pem").read_bytes(),
#(
#             Path("./src/certificates/rootCA_cert.pem").read_bytes(),
#             Path("./src/certificates/rootCA_key.pem").read_bytes() ),
             client=client)
    else:
        fl.client.start_client(
             server_address=params.server_address,
             client=client)
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************

    print(" _____________________________________________client sim  :: main :: termina")


if __name__ == "__main__":
    main()
