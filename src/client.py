import sys
import torch
import flwr as fl
import numpy as np
from typing import Dict, List, Tuple

from pathlib import Path

from collections import OrderedDict

from train import train
from test import test
from utils import Parameters
from model_wrapper import ModelWrapper

from dataloaders import MMsDataSet,LightningWrapperData

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, params):
        self.params = params

        if torch.cuda.is_available() and params.device == 'cuda':
            device = torch.device('cuda')
        else:
            device = torch.device("cpu")

        model = ModelWrapper(params)
        self.model = model.to(device)
        
        if params.dataset == "MMs":
            self.dataset = MMsDataSet(params)
        elif params.dataset == "LightningWrapperData":
            self.dataset = LightningWrapperData(params)
        else:
            print("Dataset not available")
            exit()
    
        self.dataset.setup("fit")


    def get_parameters(self, config): # config not needed at all
        print(f"[Client {self.params.client_id}] get_parameters")
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters:List[np.ndarray]):
        self.model.train()
        # Si esto del self.model.train no funciona porque no reconoce la
        # función entonces deberías sustituírla por nuestra train:
        # train(self.model,params)
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
    
    def fit(self, parameters, params):
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
