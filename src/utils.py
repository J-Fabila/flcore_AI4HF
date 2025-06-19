### ¿Utilizarmos hydra o será mejor a la vieja usanza?
import yaml

import os
import re
import glob
import argparse

class Parameters:
    def __init__(self):
        self.params = {
            # Device variables
            'device': 'cuda', # 'cuda' or 'cpu'
            'n_gpu': 1,
            'n_gpu_nodes': 1,
            'num_workers': 4,
            'client_id': None,
            'num_clients': 5,
            'set_server' : False,
            'strategy' : None,
            'min_fit_clients' : 2,
            'min_evaluate_clients' : 2,
            'min_available_clients' : 2,
            'metrics_aggregation' : None,
            'server_address' : None,

            # Model variables
            'federated' : None,
            'use_certificates' : None,
            'local_model': None,
            'features': None,
            'configuration_file' : None,
            'load_checkpoint': None,
            'task': None,
            'in_channels': None,
            'num_labels' : 1,
            'feature_size' : 1,
            'mlp_hidden_sizes' : [1,1],
            'mlp_output_size' : 1,
            'ode_hidden_size' : 1,
            'ode_num_layers' : 1,
            'ode_batch_norm' : 1,
            'time_nums': 50,

            'MLP_preprocess' : None,
            'n_classes' : None,
            'UNet_depth': None,
            'UNet_bilinear': None,
            'UNet_custom_shape':None,

            # Training variables
            'batch_size': 32,
            'train_size': 0.8,
            'val_size': 0.1,
            'test_size': 0.1,
            'epochs': 10,
            'num_rounds':10,
            'verbatim':False,

            # Optimizer variables
            'optimizer': 'Adam',
            'dropout': 0.1,
            'lr': 0.001,
            'lr_min': 1e-6,
            'lr_factor': 0.5,
            'lr_patience': 10,
            'lr_scheduler': 'ReduceLROnPlateau',
            'clip': 1.0,
            'early_stopping_patience': 10,
            
            # Data set variables
            'dataset': None,
            'dataset_root': None,
            'data_folder': None,
            'train_filepath': None,
            'test_filepath': None,
            'target_label': None,
            'n_channels' : 1,
            # integración con el cmd line
            'data_path': None,
            'data_file': None,

            # Bias mitigation
            'bias_preprocessing' : False,
            'method' : None,
            'sensitive': None,
            'label': None,

            # Logging variables
            'log_path': './',
            'every_n_epochs': 1,
            'wandb_track': False,
            'wandb_project': None,
            'wandb_run_name': None,
            'wandb_entity': None,           
            'save_top_k': 1
            }

    def GetParamsCMD(self):
        parser = argparse.ArgumentParser(description="Training configuration")
        
        # Device variables
        parser.add_argument('--device', type=str, default='cuda') #, choices=['cuda', 'cpu'])
        parser.add_argument('--n_gpu', type=int, default=1)
        parser.add_argument('--n_gpu_nodes', type=int, default=1)
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--client_id', type=int, default=None)
        parser.add_argument('--num_clients', type=int, default=5)
        parser.add_argument('--set_server', type=str, default="False") #action='store_true')
        parser.add_argument('--strategy', type=str, default=None)
        parser.add_argument('--min_fit_clients', type=int, default=2)
        parser.add_argument('--min_evaluate_clients', type=int, default=2)
        parser.add_argument('--min_available_clients', type=int, default=2)
        parser.add_argument('--metrics_aggregation', type=str, default=None)
        parser.add_argument('--server_address', type=str, default=None)
        
        # Model variables
        parser.add_argument('--federated', type=str, default="False")
        parser.add_argument('--use_certificates', type=str, default="False")
        parser.add_argument('--local_model', type=str, default=None)
        parser.add_argument('--features', type=str, default=None)
        parser.add_argument('--configuration_file', type=str, default=None)
        parser.add_argument('--load_checkpoint', type=str, default=None)
        parser.add_argument('--task', type=str, default=None)
        parser.add_argument('--in_channels', type=int, default=None)
        parser.add_argument('--num_labels', type=int, default=1)
        parser.add_argument('--feature_size', type=int, default=1)
        parser.add_argument('--mlp_hidden_sizes', type=int, nargs='+', default=[1,1])
        parser.add_argument('--mlp_output_size', type=int, default=1)
        parser.add_argument('--ode_hidden_size', type=int, default=1)
        parser.add_argument('--ode_num_layers', type=int, default=1)
        parser.add_argument('--ode_batch_norm', type=int, default=1)
        parser.add_argument('--time_nums', type=int, default=50)
        parser.add_argument('--MLP_preprocess', type=str, default=None)
        parser.add_argument('--n_classes', type=int, default=None)
        parser.add_argument('--UNet_depth', type=int, default=None)
        parser.add_argument('--UNet_bilinear', action='store_true')
        parser.add_argument('--UNet_custom_shape', type=str, default=None)
        
        # Training variables
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--train_size', type=float, default=0.8)
        parser.add_argument('--val_size', type=float, default=0.1)
        parser.add_argument('--test_size', type=float, default=0.1)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--num_rounds', type=int, default=10)
        parser.add_argument('--verbatim', action='store_true')
        
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
        parser.add_argument('--dataset', type=str, default=None)
        parser.add_argument('--dataset_root', type=str, default=None)
        parser.add_argument('--data_folder', type=str, default=None)
        parser.add_argument('--train_filepath', type=str, default=None)
        parser.add_argument('--test_filepath', type=str, default=None)
        parser.add_argument('--target_label', type=str, default=None)
        parser.add_argument('--n_channels', type=int, default=1)
        parser.add_argument('--data_path', type=str, default=None)
        parser.add_argument('--data_file', type=str, default=None)

        # Bias mitigation
        parser.add_argument('--bias_preprocessing', type=str, default=None)
        parser.add_argument('--method', type=str, default=None)
        parser.add_argument("--sensitive", type=str, nargs='+', default=None, help="Sensitive variables")
        parser.add_argument('--label',type=str, default=None)

        # Logging variables
        parser.add_argument('--log_path', type=str, default='./')
        parser.add_argument('--every_n_epochs', type=int, default=1)
        parser.add_argument('--wandb_track', type=str, default="False")
        parser.add_argument('--wandb_project', type=str, default=None)
        parser.add_argument('--wandb_run_name', type=str, default=None)
        parser.add_argument('--wandb_entity', type=str, default=None)
        parser.add_argument('--save_top_k', type=int, default=1)
        
        return parser

    def GetParams(self, input_file):
        ext = input_file.split(".")[-1]
        if "yaml" in ext or  "yml" in ext:
            with open(input_file) as f:
                params_yaml = yaml.load(f, Loader=yaml.FullLoader)
                   
                for i in self.params.keys():
                    if i in params_yaml.keys():
                        self.params[i] = params_yaml[i]

        # Device variables
        self.device = self.params["device"]
        self.n_gpu = self.params["n_gpu"]
        self.n_gpu_nodes = self.params["n_gpu_nodes"]
        self.num_workers = self.params["num_workers"]
        self.client_id = self.params["client_id"]
        self.num_clients = self.params["num_clients"]
        self.set_server = self.params["set_server"]
        self.strategy = self.params["strategy"]
        self.min_fit_clients = self.params["min_fit_clients"]
        self.min_evaluate_clients = self.params['min_evaluate_clients']
        self.min_available_clients = self.params['min_available_clients']
        self.metrics_aggregation = self.params["metrics_aggregation"]
        self.server_address = self.params['server_address']

        # Model variables
        self.federated = self.params["federated"]
        self.use_certificates = self.params["use_certificates"]
        self.local_model = self.params["local_model"]
        self.features = self.params["features"]
        self.configuration_file = self.params["configuration_file"]
        self.load_checkpoint = self.params["load_checkpoint"]
        self.task = self.params["task"]
        self.num_labels = self.params["num_labels"]
        self.feature_size = self.params["feature_size"]
        self.mlp_hidden_sizes = self.params["mlp_hidden_sizes"]
        self.mlp_output_size = self.params["mlp_output_size"]
        self.ode_hidden_size = self.params["ode_hidden_size"]
        self.ode_num_layers = self.params["ode_num_layers"]
        self.ode_batch_norm = self.params["ode_batch_norm"]
        self.time_nums = self.params["time_nums"]

        self.MLP_preprocess = ['MLP_preprocess']
        self.in_channels = self.params["in_channels"]
        self.n_classes = self.params["n_classes"]
        self.UNet_depth = self.params["UNet_depth"]
        self.UNet_bilinear = self.params["UNet_bilinear"]
        self.UNet_custom_shape = self.params["UNet_custom_shape"]

        # Training variables
        self.batch_size = self.params["batch_size"]
        self.train_size = self.params["train_size"]
        self.val_size = self.params["val_size"]
        self.test_size = self.params["test_size"]
        self.epochs = self.params["epochs"]
        self.num_rounds = self.params["num_rounds"]
        self.verbatim = self.params["verbatim"]
        
        # Optimizer variables
        self.optimizer = self.params["optimizer"]
        self.dropout = self.params["dropout"]
        self.lr = self.params["lr"]
        self.lr_min = self.params["lr_min"]
        self.lr_factor = self.params["lr_factor"]
        self.lr_patience = self.params["lr_patience"]
        self.lr_scheduler = self.params["lr_scheduler"]
        self.clip = self.params["clip"]
        self.early_stopping_patience = self.params["early_stopping_patience"]

        # Data set variables
        self.dataset = self.params["dataset"]
        self.dataset_root = self.params["dataset_root"]
        self.data_folder = self.params["data_folder"]
        self.train_filepath = self.params["train_filepath"]
        self.test_filepath = self.params["test_filepath"]
        self.target_label = self.params["target_label"]
        self.n_channels = self.params["n_channels"]
        # integración con el cmd line
        self.data_path = self.params["data_path"]
        self.data_file = self.params["data_file"]
        
        # Bias mitigation
        self.bias_preprocessing = self.params["bias_preprocessing"]
        self.method = self.params["method"]
        self.sensitive = self.params["sensitive"]
        self.label = self.params["label"]

        # Logging variables
        self.log_path = self.params["log_path"]
        self.every_n_epochs = self.params["every_n_epochs"]
        self.wandb_track = self.params["wandb_track"]
        self.wandb_project = self.params["wandb_project"]
        self.wandb_run_name = self.params["wandb_run_name"]
        self.wandb_entity = self.params["wandb_entity"]
        self.save_top_k = self.params["save_top_k"]
        self.ParametersVerifier()
    
    def ParametersVerifier(self):
        if self.client_id == "None":
            print(" Client ID not selected")
            if self.set_server == False:
                print(" Server was not selected")
                print(" You have to set either client or server")
                exit()
        if self.client_id != "None":
            if self.set_server == True:
                print(" You selected both client and server")
                exit()



def get_parameters(directory):
    ckpt_files = glob.glob(os.path.join(directory, '*.ckpt'))

    best_checkpoint = None
    best_val_loss = float('inf')

    for ckpt_file in ckpt_files:
        match = re.search(r'epoch=(\d+)-val_loss=([\d.]+)\.ckpt', ckpt_file)
        if match:
            epoch = int(match.group(1))
            val_loss = float(match.group(2))
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_checkpoint = ckpt_file
    # Prepares filename for the next iteration
    new_filename = "model_client_0_round_0.ckpt"
    os.rename(best_checkpoint, os.path.join(directory,  new_filename))
