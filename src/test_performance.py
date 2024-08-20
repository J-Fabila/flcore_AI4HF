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
print("dataset", params.dataset_root)
dataset = torch.load(params.dataset_root)

acc = 0.0
correct = 0
total = 0
test_sampling_size=100
print("inicia")
with torch.no_grad():
    loss = 0.0
    for i in range(test_sampling_size):
        images, target = dataset[i]["image"], dataset[i][params.target_label]
        print(" shape ",images.unsqueeze(0).shape, target)
        outputs = model(images.unsqueeze(0).to(device))
        print(" sale model", target,target.shape)
        print(" output", outputs, outputs.shape)
        _, predicted = torch.max(outputs.data, 1)
        total += 1 #target.size(0)
        correct += (predicted == target).sum().item()
    acc = 100 * correct // total

print(f'Accuracy of the network on test images is: {100 * correct // total} %')
