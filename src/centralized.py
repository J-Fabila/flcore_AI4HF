import sys
import torch
import numpy as np
from typing import Dict, List, Tuple

from train import train
from test import test
from utils import Parameters
from model_wrapper import ModelWrapper
from dataloaders import MMsDataSet,LightningWrapperData

params = Parameters()
config_file = sys.argv[1]
params.GetParams(config_file)

if torch.cuda.is_available() and params.device == 'cuda':
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

model = ModelWrapper(params)
model = model.to(device)
if params.dataset == "MMs":
    dataset = MMsDataSet(params)
elif params.dataset == "LightningWrapperData":
    dataset = LightningWrapperData(params)
else:
    print("Dataset not available")
    exit()
dataset.setup("fit")

train(model,params,dataset)
