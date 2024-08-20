import sys

import torch

from utils import Parameters
from model_wrapper import ModelWrapper
from dataloaders import LightningWrapperData

print("ARGV[1] = config file")
config_file =  sys.argv[1]

#if len(sys.argv) > 1:
#    params = Parameters()
#    config_file = sys.argv[1]
#    params.GetParams(config_file)
#else:
#    print("*.yaml input file needed as command line argument")
#    exit()

params = Parameters()
#config_file = sys.argv[1]
params.GetParams(config_file)

model = ModelWrapper(params)
model.load(params.load_checkpoint)

dataset = LightningWrapperData(params)
dataset.setup("train")
testing = dataset.test_dataloader()

predictions = []
inputs = []
targets = []
contador = 0
with torch.no_grad():
    if params.task == "Classification":
        total = 0
        correct = 0
        for batch in testing:
            inputs = batch["image"]
            target = batch[params.target_label]
            outputs = model.forward(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        acc = 100 * correct // total       
        print("acc ",acc, "total" , total)
    elif (params.task == "Segmentation"):
        for batch in testing:
            input = batch["image"]
            target = batch[params.target_label]
            pred = model.forward(input)
#            print(" SHAPES DEL BATCH")
#            print("inputs",input.shape)
#            print("target",target.shape)
            print("pred", pred.shape, pred.max())
            inputs.append(input)
            targets.append(target)
            predictions.append(pred)
            contador = contador +1
            if contador > 10:
                break
    elif (params.task == "Regression"):
        pass

_pred = torch.cat(predictions, dim=0)
all_pred = torch.argmax(_pred,dim=1)
all_in = torch.cat(inputs,dim=0)
all_tar = torch.cat(targets, dim=0)
data = {"predictions":all_pred, "all_inputs":all_in, "all_targets":all_tar}
# Temporalmente queda algo cutre pero después agrégale una os.path join y tal
# Alternativamente podríamos agregar otra variable que tenga libertad de elegir
# el nombre pero ya me parece un poco innecesario para usar con params.log_path
torch.save(data, "prediction.pt")
