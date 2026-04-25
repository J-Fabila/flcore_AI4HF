import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
#from torchmetrics.classification import Dice
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Models.BasicConvolutional import BasicConvolutional
from Models.UNet import UNet
from Models.ResNet import *
#from Models.MLP.MLP import MLP_SODEN
#from Models.MLP.metrics import precision_test, cindex 

from binary import hd95, hd, assd, jc

class ModelWrapper(pl.LightningModule):
    def __init__(self, config):
        super(ModelWrapper, self).__init__()
        self.config = config
        self.task = config['task']
        if self.config['local_model'] == "Basic":
            self.local_model = BasicConvolutional(config)
        if self.config['local_model'] == "UNet":
            self.local_model = UNet(config)
        elif self.config['local_model'] == "Resnet18":
            self.local_model = resnet18(config)
        elif self.config['local_model'] == "Resnet34":
            self.local_model = resnet34(config)
        elif self.config['local_model'] == "Resnet50":
            self.local_model = resnet50(config)
        elif self.config['local_model'] == "Resnet101":
            self.local_model = resnet101(config)
        elif self.config['local_model'] == "Resnet152":
            self.local_model = resnet152(config)
            # Quantized ResNet
        else:
            print("No option selected")

        self.criterion = nn.CrossEntropyLoss() if self.config['n_classes'] > 1 else nn.BCEWithLogitsLoss()
        if (self.config['task'] == "Segmentation"):
            self.dice = Dice(average='micro')
        self.save_hyperparameters(self.config)

    def forward(self, image):
        #orward(self, inputs: Dict[str, torch.Tensor], label: torch.Tensor, full_eval: bool = False) -> Tuple[list, torch.Tensor]:
        inputs , label, full_eval = image[0], image[1], image[2]
        return self.local_model.forward(inputs, label, full_eval)
    
    def step(self, batch, stage):
        loss = 0.0
        images, target = batch["image"], batch[self.config['target_label']]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        with torch.set_grad_enabled(True):
            if (self.config['task'] == "Classification"):
                loss += F.cross_entropy(outputs, target)
            elif (self.config['task'] == "Segmentation"):
                _, _, d1 , d2 = target.shape
                original_shape = (batch_size, d1,d2,self.config['n_classes'])
                flat = target.flatten()
                encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.config['n_classes'])
                target = encoded.view(original_shape)
                outputs_prob = torch.sigmoid(outputs)
                target_prob = target.permute(0, 3, 1, 2).float()
                loss += F.binary_cross_entropy(outputs_prob, target_prob)
                loss += self.dice(outputs_prob,target_prob.to(int))
            elif (self.config['task'] == "Regression"):
                pass
        return loss.to(dtype = torch.float32)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch,"train")
        logs = {'train_loss': loss}
        self.log("train_loss", loss,batch_size=self.config['batch_size']) #,prog_bar=False, on_step=False)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss = self.step( batch, "val")
        images, target = batch["image"], batch[self.config['target_label']]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        if self.config['task'] == "Classification":
            acc = 0.0
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = 100 * correct // total            
            self.log_dict({"val_loss":loss,"acc": acc})
            logs = {'val_loss': loss,'acc':acc}
        elif (self.config['task'] == "Segmentation"):
            _, _, d1 , d2 = target.shape
            original_shape = (batch_size, d1,d2,self.config['n_classes'])
            flat = target.flatten()
            encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.config['n_classes'])
            target = encoded.view(original_shape)
            target_prob = target.permute(0, 3, 1, 2).to(int)
            ################################################################
            outputs_prob = torch.sigmoid(outputs)
            dice_ = self.dice(outputs_prob,target_prob)
            ################################################################
            _, predi = torch.max(outputs_prob, 1)
            _, outsi = torch.max(target_prob,1)
            
            encoded = torch.nn.functional.one_hot(predi.flatten().to(torch.int64),self.config['n_classes'])
            output = encoded.view(original_shape)
            outputs_prob = output.permute(0, 3, 1, 2).to(int)

            hausdorff = 0.0
            for clase in range(self.config['n_classes']):
                #print("CLASE ", clase, target_prob[:,clase,:,:].max() , outputs_prob[:,clase,:,:].max())
                #print(" TARGETS PROB", target_prob[0,:,0,0])
                #print(" DIMENSIONES outputs prob", outputs_prob[0,:,0,0])
            
                if target_prob[:,clase,:,:].max() == 0:
                    target_prob[0,clase,0,0] = 1   
                if outputs_prob[:,clase,:,:].max() == 0:
                    outputs_prob[0,clase,0,0] = 1    
                hausdorff += hd(outputs_prob[:,clase,:,:].cpu().numpy(),target_prob[:,clase,:,:].cpu().numpy())
                print("HAUSDORFF ",clase, hausdorff)
            hausdorff = hausdorff/self.config['n_classes']
            print("HAUSDORFF ", hausdorff)
            logs = {'val_loss': loss,"dice":dice_,"hausdorff":hausdorff}
            self.log_dict({"val_loss":loss,"dice":dice_,"hausdorff":hausdorff})
        return {"loss" : loss, 'log' : logs}

    def test_step(self, batch, batch_idx):
        loss = self.step( batch, "test")
        images, target = batch["image"], batch[self.config['target_label']]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        if self.config['task'] == "Classification":
            acc = 0.0
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = 100 * correct // total            
            self.log_dict({"test_loss":loss,"acc": acc})
            logs = {'test_loss': loss,'acc':acc}
        
        elif (self.config['task'] == "Segmentation"):
            _, _, d1 , d2 = target.shape
            original_shape = (batch_size, d1,d2,self.config['n_classes'])
            flat = target.flatten()
            encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.config['n_classes'])
            target = encoded.view(original_shape)
            target_prob = target.permute(0, 3, 1, 2).to(int)
            ################################################################
            outputs_prob = torch.sigmoid(outputs)
            dice_ = self.dice(outputs_prob,target_prob)
            ################################################################
            _, predi = torch.max(outputs_prob, 1)
            _, outsi = torch.max(target_prob,1)
            
            encoded = torch.nn.functional.one_hot(predi.flatten().to(torch.int64),self.config['n_classes'])
            output = encoded.view(original_shape)
            outputs_prob = output.permute(0, 3, 1, 2).to(int)
   
            hausdorff = 0.0
            for clase in range(self.config['n_classes']):
                if target_prob[:,clase,:,:].max() == 0:
                    target_prob[0,clase,0,0] = 1
                if outputs_prob[:,clase,:,:].max() == 0:
                    outputs_prob[0,clase,0,0] = 1
                hausdorff += hd(outputs_prob[:,clase,:,:].cpu().numpy(),target_prob[:,clase,:,:].cpu().numpy())
                print("HAUSDORFF clase",clase, hausdorff)
            hausdorff = hausdorff/self.config['n_classes']
            print("HAUSDORFF ", hausdorff)
            logs = {'val_loss': loss,"dice":dice_,"hausdorff":hausdorff}
            self.log_dict({"val_loss":loss,"dice":dice_,"hausdorff":hausdorff})
        return {"loss" : loss, 'log' : logs}
        
        ###################################################
        #def training_step(self, batch, batch_idx):
        #self.log("my_metric", x)
        # or a dict to log all metrics at once with individual plots
        #def training_step(self, batch, batch_idx):
        #self.log_dict({"acc": acc, "recall": recall})
        # Esto muy probablemente se tendría que incluir en
        # algún punto, aunque quizás no ahora
        #preds = torch.argmax(outputs, dim=1)
        #correct = torch.sum(preds == labels)
        #return {"test_loss": loss, "correct": correct, "total": len(labels)}
        ###################################################

    def configure_optimizers(self):
        optim_dict = {
            "SGD": torch.optim.SGD(self.parameters(), lr=self.config['lr'], momentum=0.9),
            "Adam": torch.optim.Adam(self.parameters(), lr=self.config['lr']),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=self.config['lr']),
            "Adagrad": torch.optim.Adagrad(self.parameters(), lr=self.config['lr']),
            "Adadelta": torch.optim.Adadelta(self.parameters(), lr=self.config['lr']),
            "Adamax": torch.optim.Adamax(self.parameters(), lr=self.config['lr']),
            "ASGD": torch.optim.ASGD(self.parameters(), lr=self.config['lr'])
        }
        optimizer = optim_dict[self.config['optimizer']]
        ################################################
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config['lr'])

        ################################################
        scheduler_dict = {
            "ReduceLROnPlateau": ReduceLROnPlateau(
                optimizer,
                "min",
                factor=float(self.config['lr_factor']),
                patience=float(self.config['lr_patience']),
                min_lr=float(self.config['lr_min']),
            ),
            "StepLR": StepLR(optimizer, step_size=1, gamma=self.config['lr_factor']),
            "MultiStepLR": MultiStepLR(
                optimizer, milestones=[1, 2, 3], gamma=self.config['lr_factor']
            ),
            "ExponentialLR": ExponentialLR(optimizer, gamma=self.config['lr_factor']),
            "CosineAnnealingLR": CosineAnnealingLR(
                optimizer, T_max=10, eta_min=float(self.config['lr_min'])
            ),
            "CyclicLR": CyclicLR(
                optimizer,
                base_lr=self.config['lr_min'],
                max_lr=self.config['lr'],
                step_size_up=10,
                cycle_momentum=False,
            ),
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=float(self.config['lr_min'])
            ),
        }

        lr_scheduler = {
            "scheduler": scheduler_dict[self.config['lr_scheduler']],
            "monitor": 'val_loss',
            "interval": "epoch",
            "frequency": 1,
        }
        #return {"optimizer": optimizer, "lr_scheduler": lr_scheduler, "monitor": "val_loss"}
        return [optimizer], [lr_scheduler]

    def optimizer_step(self, *args, **kwargs):
        optimizer = self.optimizers()
        super().optimizer_step(*args, **kwargs)
        optimizer.zero_grad()

    def load(self,input_path):
        # Tenemos que verificar la compatibilidad entre dispositivos
        checkpoint = torch.load(input_path, map_location="cpu")

        self.load_state_dict(checkpoint["state_dict"])
