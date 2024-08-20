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

data_set = []
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
            id = batch["id"]
            pixel_spacing = batch["pixel_spacing"]
            original_size_0 = batch["original_size"][-2].item()
            original_size_1 = batch["original_size"][-1].item()
            original_size = [original_size_0, original_size_1]
#            print("original size",original_size)
            input = batch["image"]
            pred = model.forward(input)
            print("PRED SHAPE", pred.shape)
########################## AQUI TODO EL PROBLEMA
#_pred = torch.cat(predictions, dim=0)
            pred_ = torch.argmax(pred,dim=1)
#################################################
#            print(" SHAPES DEL BATCH")
#            print("inputs",input.shape)
#            print("target",target.shape)
            print("pred", pred_.shape,pred_.max())
#            targets.append(target)
            contador = contador +1
            temp={"id":id,"image": input, "mask": pred_,"pixel_spacing":pixel_spacing, "original_size": original_size}
            data_set.append(temp)

            if contador > 10:
                break
    elif (params.task == "Regression"):
        pass

# Temporalmente queda algo cutre pero después agrégale una os.path join y tal
# Alternativamente podríamos agregar otra variable que tenga libertad de elegir
# el nombre pero ya me parece un poco innecesario para usar con params.log_path
torch.save(data_set, "etiquetadas.pt")
