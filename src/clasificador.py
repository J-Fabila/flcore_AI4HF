import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Models.BasicConvolutional import BasicConvolutional
from Models.UNet import UNet
from Models.ResNet import *
from utils import Parameters
from model_wrapper import ModelWrapper

if len(sys.argv) > 1:
    params = Parameters()
    config_file = sys.argv[1]
    imagen = sys.argv[2]
    params.GetParams(config_file)
else:
    print("*.yaml input file needed as command line argument")
    exit()

if torch.cuda.is_available() and params.device == 'cuda':
    device = torch.device('cuda')
else:
    device = torch.device("cpu")

model = ModelWrapper(params)
model = model.to(device)

params.load_checkpoint = "/home/jorge/work_dir/federado/Federado/flower_version/models/model_españa.ckpt"

if params.load_checkpoint != "None":
    model.load(params.load_checkpoint)

Aqui copia las  parte del preprocessing que lee las imagenes, las convierte en tensores, transforma
y luego alimentamos al modelo con FileExistsError

clase = 0.0
print("Image given is classified as:",clase)