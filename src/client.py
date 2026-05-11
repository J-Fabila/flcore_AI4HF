import os
import sys
import time
import glob
import torch
import flwr as fl
import numpy as np
import argparse
from typing import Dict, List, Tuple

from pathlib import Path

from collections import OrderedDict

from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim

from train import train
from test import test

from mlp_utils import load_json_config
from data_mlp import process_imputed_data
from Models.MLP.model import MLP_SODEN
from mlp_trainer import main_training_loop

#from model_wrapper import ModelWrapper
from dataloaders import MMsDataSet,LightningWrapperData  

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, config):
        self.config = config
        if torch.cuda.is_available() and config['device'] == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")
        self.device = device
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
            if config['features'] == "maggic":
                config_selector = 0
            elif config['features'] == "maggic_plus":
                config_selector = 1
# *************** AQUI EN VEZ DE SUSTITUIR EL DICCIOARNIO AÑADIR NUEVAS KEYS Y PUNTO
            self.config.update(configurations[config_selector])
            #Aqui primer problema: esto no debería estar alambrado
            #self.model_folder = '/home/jorge/work_dir/nouman/AI4HF-OXF-Modelling/new_models'
            # o sería mejor ponerlo como variable
            self.model_folder = config['log_path'] #'data'
            os.makedirs(self.model_folder, exist_ok=True)  # Crea el directorio si no existe
            self.model_file = os.path.join(self.model_folder, f"{self.config['features']}_model.pth")
            model = MLP_SODEN(self.config)
            model.suffix = self.config['features']

# ==================================================================================
            torch.save(model.state_dict(), f'{self.model_folder}/{model.suffix}_model.pth')
# ==================================================================================
            self.model = model.to(device)

#            model.suffix = self.config['features']
            self.optimizer = optim.Adam(model.parameters(), lr=1e-3)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=config['lr_patience'], verbose=True)

        if config['model'] == "MLP":
            if config['MLP_preprocess'] == "True":
            #************* * * *  *  *  *   *  Data preprocessing  *    *  *  *  *  * * *************
                configuration = load_json_config(config['configuration_file'])
                # AQUI SE MODIFICAN LAS VARS SI HACEN FALTA
                configuration["data_folder"] = config['data_folder']
                pattern = "*.parquet"
                parquet_files = glob.glob(os.path.join(configuration["data_folder"], pattern))
                if len(parquet_files) == 0:
                    print("No parquet files found in ",configuration["data_folder"])
                    sys.exit()

                nombre_archivo = parquet_files[-1]
                configuration["file_path"] = os.path.join(config['data_folder'], nombre_archivo)
                # Process imputed data
                process_imputed_data(configuration)
            #************* * * *  *  *  *   *   *      *     *     *    *  *  *  *  * * *************

    def get_parameters(self, config): # config not needed at all
        if self.config['model'] == "MLP":
            loc_model = torch.load(self.model_file)
            print("regresa valores")
            #print([val.cpu().numpy() for  val in loc_model.values()])
            return [val.cpu().numpy() for  val in loc_model.values()]
        else:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters:List[np.ndarray]):
        print(f"[Client {self.config['client_id']}] set_parameters")
        if self.config['model'] == "MLP":
            # # fFalta sacar el model_keys de algun lado           
            params_dict = zip(self.model_keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            # tenemos algo asi? quizas lo mas facil sea escribir un archivo pth y 
            # cargar el archivo com model load
            print("SET PARAMETERS::LOAD STATE DICT")
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
        if self.config['model'] == "MLP":
            train_filepath = os.path.join(self.config['data_folder'], f"train_{self.config['features']}.pt")
            test_filepath = os.path.join(self.config['data_folder'], f"valid_{self.config['features']}.pt")
            model_path = self.model_folder # el directorio de log de los params
            print("INICIA MAIN TRAIING LOOP")
            # Idea: self.results # asi almacenamos los results que son:
            # main_training_loop :: return tempprc, tempauroc, test_loss, model
            # así imprimimos los valores en  la línea 168
            results = main_training_loop(self.model, train_filepath, test_filepath,
                                    model_path, self.optimizer, self.config['epochs'],
                                   self.config['lr_patience'], self.scheduler, self.device)
            ## AQUI TAMBIEN
            self.tempprc , self.tempauroc, self.test_loss, _ = results

            trainloader_dataset_len = 1.0 #self.dataset.train_size
            #print("FIT :: PREVIO A REGRESAR PARAMS",self.get_parameters(config={}),trainloader_dataset_len)                
            return self.get_parameters(config={}), int(trainloader_dataset_len), {}
        else:
            print(" ***************************************** FIT self.config.client_id ", self.config)
            print(f"[Client {self.config['client_id']}] fit")
            self.set_parameters(parameters)
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
            train(self.model,self.config,self.dataset)
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
        if self.config['model'] == "MLP":
            #return tempprc, tempauroc, test_loss, model
#            return 0.0, 1, {"accuracy":0.0}
#           OJO que tendrías que cambiar las variables en el server: linea 14 en la definicion del weighted avarage
#           tendrias que adaptar la funcion
            return 0.0, 1, {"tempprc":self.tempprc, "tempauroc":self.tempauroc,"loss":self.test_loss}
        else:
            # parameters es una lista y params un diccionario vacio
            # En principio aqui aceptamos params, pero no depende de nosotros pasar params,
            # flower pasa los parametros que le salen de los huevos
            print(f"^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^[Client {self.config['client_id']}] evaluate")
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

if __name__ == "__main__":
    ## OJO: aqui falta cambiar el len(dataset) en evaluate y fit
    print(" MAIN DEL CLIENTE =======================================")
    print(" client sim  :: main :: inicia")
# ____________________________________________________________________________________-
    parser = argparse.ArgumentParser(description="Training configuration")
    
    # Device variables
    parser.add_argument('--device', type=str, default='cuda') #, choices=['cuda', 'cpu'])
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--n_gpu_nodes', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--client_id', type=str, default="client")
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--production_mode', type=str, default="False")
    parser.add_argument("--local_port", type=str, default="8081")
    parser.add_argument('--server_address', type=str, default=None)
    parser.add_argument("--sandbox_path", type=str, default="./sandbox", help="Sandbox path to use")

    # Model variables
    parser.add_argument('--model', type=str, default="MLP")
    parser.add_argument('--task', type=str, default="classification")
    parser.add_argument('--features', type=str, default="maggic")
    parser.add_argument('--configuration_file', type=str, default="configs/sample_config.json")
    parser.add_argument('--MLP_preprocess', type=str, default="True")

    # Training variables
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--train_size', type=float, default=0.8)
    parser.add_argument('--val_size', type=float, default=0.1)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--num_rounds', type=int, default=10)
    parser.add_argument('--verbatim', type=str, default="False")
    
    # Optimizer variables
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_min', type=float, default=1e-6)
    parser.add_argument('--lr_factor', type=float, default=0.5)
    parser.add_argument('--lr_patience', type=int, default=10)
    parser.add_argument('--lr_scheduler', type=str, default='ReduceLROnPlateau')
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--early_stopping_patience', type=int, default=10)
    
    # Dataset variables
    parser.add_argument('--data_folder', type=str, default=None)
    
    # Logging variables
    parser.add_argument('--log_path', type=str, default='./')
    parser.add_argument('--every_n_epochs', type=int, default=1)
    parser.add_argument('--wandb_track', type=str, default="False")
    parser.add_argument('--wandb_project', type=str, default=None)
    parser.add_argument('--wandb_run_name', type=str, default=None)
    parser.add_argument('--wandb_entity', type=str, default=None)
    parser.add_argument('--save_top_k', type=int, default=1)

    args = parser.parse_args()
    config = vars(args)
# ____________________________________________________________________________________-

    if config["production_mode"] == "True":
        node_name = os.getenv("NODE_NAME")
        data_path = os.getenv("DATA_FOLDER")
        central_ip = os.getenv("FLOWER_CENTRAL_SERVER_IP")
        central_port = os.getenv("FLOWER_CENTRAL_SERVER_PORT")
        sandbox_path = os.getenv("SANDBOX_PATH")

        flower_ssl_cacert = os.getenv("FLOWER_SSL_CACERT")
        root_certificate = Path(f"{flower_ssl_cacert}").read_bytes()

    else:
        print("**********************************PROD MODE FALSE")
        print("CONFIG LOCAL PORT", config["local_port"])
        node_name = config["client_id"]
        data_path = config["data_folder"]
        central_ip = "LOCALHOST"
        central_port = config["local_port"]
        sandbox_path = config["sandbox_path"]
        root_certificate = None

    # Create sandbox log file path
    sandbox_log_file = Path(os.path.join(config["sandbox_path"], "log_client.txt"))
    
    # Creation of the model instances
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************
    # Aquí empieza lo interesante. Lo demás está en principio listo.
    # Ten cuidado de generar los clientes con cuidado, sobre todo los
    # get parameters y set parameters y que sea consistente con la integración pl
    #************* * * *  *  *  *   *    *    *  *  *  *  * * *************

    # No needed to load data since that will be done by training torch lightning wrapper
    client = FlowerClient(config).to_client()
    fl.client.start_client(
        server_address=f"{central_ip}:{central_port}",
        # credentials=ssl_credentials,
        root_certificates=root_certificate,
        client=client,
        #channel=channel,
    )
    print(" ________________________________________________  Cliente generado")
    """
    for attempt in range(3):
        try:
            if isinstance(client, fl.client.NumPyClient):
                print("IS INSTANCE OF NUMPY CLIENT")
                fl.client.start_numpy_client(
                    server_address=f"{central_ip}:{central_port}",
                    #credentials=ssl_credentials,
                    root_certificates=root_certificate,
                    client=client,
                    #channel=channel,
                )
            else:
                print("***************ELSE")
                print("CENTRAL IP", central_ip)
                print("CENTRAL PORT", central_port)
                print("ROOT CERT", root_certificate)
                print("_________________________________")
                fl.client.start_client(
                    server_address=f"{central_ip}:{central_port}",
                    # credentials=ssl_credentials,
                    root_certificates=root_certificate,
                    client=client,
                    #channel=channel,
                )
            break  # Si todo salió bien, salimos del bucle
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < 2:
                time.sleep(2)  # Espera un poco antes de reintentar
            else:
                print("All connection attempts failed.")
                raise
    """
    print(" _____________________________________________client sim  :: main :: termina")
