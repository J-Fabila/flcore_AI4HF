import sys
import torch
import numpy as np
from typing import Dict, List, Tuple

from train import train
from test import test
from utils import Parameters
from model_wrapper import ModelWrapper
from dataloaders import MMsDataSet,LightningWrapperData, MLPWrapperData

params = Parameters()
config_file = sys.argv[1]
params.GetParams(config_file)

if torch.cuda.is_available() and params.device == 'cuda':
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

model = ModelWrapper(params)
model = model.to(device)
print("FLOWERCLIENT::INIT::PARAMS_DATASET",params.dataset,params.local_model)
if params.dataset == "MMs":
    dataset = MMsDataSet(params)
elif params.local_model == "MLP":
    if params.MLP_preprocess:
    #************* * * *  *  *  *   *  Data preprocessing  *    *  *  *  *  * * *************
    #     · · ·
    #     · · ·
    #     · · ·
    #************* * * *  *  *  *   *   *      *     *     *    *  *  *  *  * * *************
        pass
    else:
        pass
    dataset = MLPWrapperData(params)
elif params.dataset == "LightningWrapperData":
    dataset = LightningWrapperData(params)
else:
    print("Dataset not available")
    exit()
dataset.setup("fit")

train(model,params,dataset)
