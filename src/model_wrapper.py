import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.classification import Dice
import pytorch_lightning as pl

from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CyclicLR
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from Models.BasicConvolutional import BasicConvolutional
from Models.UNet import UNet
from Models.ResNet import *
from binary import hd95, hd, assd, jc

class ModelWrapper(pl.LightningModule):
    def __init__(self, params):
        super(ModelWrapper, self).__init__()
        self.params = params
        self.task = params.task
        if self.params.local_model == "Basic":
            self.local_model = BasicConvolutional(params)
        elif self.params.local_model == "UNet":
            self.local_model = UNet(params)
        elif self.params.local_model == "Resnet18":
            self.local_model = resnet18(params)
        elif self.params.local_model == "Resnet34":
            self.local_model = resnet34(params)
        elif self.params.local_model == "Resnet50":
            self.local_model = resnet50(params)
        elif self.params.local_model == "Resnet101":
            self.local_model = resnet101(params)
        elif self.params.local_model == "Resnet152":
            self.local_model = resnet152(params)
            # Quantized ResNet
        else:
            print("No option selected")

        self.criterion = nn.CrossEntropyLoss() if self.params.n_classes > 1 else nn.BCEWithLogitsLoss()
        if (self.params.task == "Segmentation"):
            self.dice = Dice(average='micro')
        self.save_hyperparameters(self.params.params)

    def forward(self, image):
        return  self.local_model(image)

    def step(self, batch, stage):
        loss = 0.0
        images, target = batch["image"], batch[self.params.target_label]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        with torch.set_grad_enabled(True):
            if (self.params.task == "Classification"):
                loss += F.cross_entropy(outputs, target)
            elif (self.params.task == "Segmentation"):
                _, _, d1 , d2 = target.shape
                original_shape = (batch_size, d1,d2,self.params.n_classes)
                flat = target.flatten()
                encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.params.n_classes)
                target = encoded.view(original_shape)
                outputs_prob = torch.sigmoid(outputs)
                target_prob = target.permute(0, 3, 1, 2).float()
                loss += F.binary_cross_entropy(outputs_prob, target_prob)
                loss += self.dice(outputs_prob,target_prob.to(int))
            elif (self.params.task == "Regression"):
                pass
        return loss.to(dtype = torch.float32)

    def training_step(self, batch, batch_idx):
        loss = self.step(batch,"train")
        logs = {'train_loss': loss}
        self.log("train_loss", loss,batch_size=self.params.batch_size) #,prog_bar=False, on_step=False)
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, batch_idx):
        loss = self.step( batch, "val")
        images, target = batch["image"], batch[self.params.target_label]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        if self.params.task == "Classification":
            acc = 0.0
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = 100 * correct // total            
            self.log_dict({"val_loss":loss,"acc": acc})
            logs = {'val_loss': loss,'acc':acc}
        elif (self.params.task == "Segmentation"):
            _, _, d1 , d2 = target.shape
            original_shape = (batch_size, d1,d2,self.params.n_classes)
            flat = target.flatten()
            encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.params.n_classes)
            target = encoded.view(original_shape)
            target_prob = target.permute(0, 3, 1, 2).to(int)
            ################################################################
            outputs_prob = torch.sigmoid(outputs)
            dice_ = self.dice(outputs_prob,target_prob)
            ################################################################
            _, predi = torch.max(outputs_prob, 1)
            _, outsi = torch.max(target_prob,1)
            
            encoded = torch.nn.functional.one_hot(predi.flatten().to(torch.int64),self.params.n_classes)
            output = encoded.view(original_shape)
            outputs_prob = output.permute(0, 3, 1, 2).to(int)
   
            hausdorff = 0.0
            for clase in range(self.params.n_classes):
                #print("CLASE ", clase, target_prob[:,clase,:,:].max() , outputs_prob[:,clase,:,:].max())
                #print(" TARGETS PROB", target_prob[0,:,0,0])
                #print(" DIMENSIONES outputs prob", outputs_prob[0,:,0,0])
            
                if target_prob[:,clase,:,:].max() == 0:
                    target_prob[0,clase,0,0] = 1   
                if outputs_prob[:,clase,:,:].max() == 0:
                    outputs_prob[0,clase,0,0] = 1    
                hausdorff += hd(outputs_prob[:,clase,:,:].cpu().numpy(),target_prob[:,clase,:,:].cpu().numpy())
                print("HAUSDORFF ",clase, hausdorff)
            hausdorff = hausdorff/self.params.n_classes
            print("HAUSDORFF ", hausdorff)
            logs = {'val_loss': loss,"dice":dice_,"hausdorff":hausdorff}
            self.log_dict({"val_loss":loss,"dice":dice_,"hausdorff":hausdorff})
        return {"loss" : loss, 'log' : logs}

    def test_step(self, batch, batch_idx):
        loss = self.step( batch, "test")
        images, target = batch["image"], batch[self.params.target_label]
        outputs = self(images)
        batch_size = batch["image"].shape[0]
        if self.params.task == "Classification":
            acc = 0.0
            correct = 0
            total = 0
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            acc = 100 * correct // total            
            self.log_dict({"test_loss":loss,"acc": acc})
            logs = {'test_loss': loss,'acc':acc}
        
        elif (self.params.task == "Segmentation"):
            _, _, d1 , d2 = target.shape
            original_shape = (batch_size, d1,d2,self.params.n_classes)
            flat = target.flatten()
            encoded = torch.nn.functional.one_hot(flat.to(torch.int64),self.params.n_classes)
            target = encoded.view(original_shape)
            target_prob = target.permute(0, 3, 1, 2).to(int)
            ################################################################
            outputs_prob = torch.sigmoid(outputs)
            dice_ = self.dice(outputs_prob,target_prob)
            ################################################################
            _, predi = torch.max(outputs_prob, 1)
            _, outsi = torch.max(target_prob,1)
            
            encoded = torch.nn.functional.one_hot(predi.flatten().to(torch.int64),self.params.n_classes)
            output = encoded.view(original_shape)
            outputs_prob = output.permute(0, 3, 1, 2).to(int)
   
            hausdorff = 0.0
            for clase in range(self.params.n_classes):
                if target_prob[:,clase,:,:].max() == 0:
                    target_prob[0,clase,0,0] = 1
                if outputs_prob[:,clase,:,:].max() == 0:
                    outputs_prob[0,clase,0,0] = 1
                hausdorff += hd(outputs_prob[:,clase,:,:].cpu().numpy(),target_prob[:,clase,:,:].cpu().numpy())
                print("HAUSDORFF clase",clase, hausdorff)
            hausdorff = hausdorff/self.params.n_classes
            print("HAUSDORFF ", hausdorff)
            logs = {'val_loss': loss,"dice":dice_,"hausdorff":hausdorff}
            self.log_dict({"val_loss":loss,"dice":dice_,"hausdorff":hausdorff})
        return {"loss" : loss, 'log' : logs}
        
    """def validation_step(self, batch, batch_idx):
        loss = self.step( batch, "validation")
        logs = {'validation_loss': loss}
        #---------------------------------------------------------
        #self.log("test_loss",loss,batch_size=self.params.batch_size) #, prog_bar=True, on_step=False)
        acc = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            loss = 0.0
            ################################ CUIDADO AQUI CON EL LABEL
            images, labels = batch["image"], batch["label"]
            outputs = self(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        acc = 100 * correct // total            
        #for data in testloader:
        #        images, labels = data
                # calculate outputs by running images through the network
        #        outputs = net(images)
                # the class with the highest energy is what we choose as prediction

        #print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
        self.log_dict({"test_loss":loss,"acc": acc})
        #---------------------------------------------------------
        return {"loss":loss,'log' : logs}
    
    def test_step(self, batch, batch_idx):
        loss = self.step( batch, "test")
        logs = {'test_loss': loss}
        if self.params.task == "Classification":
            acc = 0.0
            correct = 0
            total = 0
            with torch.no_grad():
                images, target = batch["image"], batch[self.params.target_label]
                outputs = self(images)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            acc = 100 * correct // total       
            self.log_dict({"test_loss":loss,"acc": acc})
        elif (self.params.task == "Segmentation"):
            self.log_dict({"test_loss":loss})
        return {"loss":loss,'log' : logs}"""

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

    """def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        total_correct = torch.stack([x["correct"] for x in outputs]).sum()
        total_items = torch.stack([x["total"] for x in outputs]).sum()
        accuracy = total_correct.float() / total_items
        self.log("test_loss", avg_loss)
        self.log("test_accuracy", accuracy)"""


    def configure_optimizers(self):
        optim_dict = {
            "SGD": torch.optim.SGD(self.parameters(), lr=self.params.lr, momentum=0.9),
            "Adam": torch.optim.Adam(self.parameters(), lr=self.params.lr),
            "RMSprop": torch.optim.RMSprop(self.parameters(), lr=self.params.lr),
            "Adagrad": torch.optim.Adagrad(self.parameters(), lr=self.params.lr),
            "Adadelta": torch.optim.Adadelta(self.parameters(), lr=self.params.lr),
            "Adamax": torch.optim.Adamax(self.parameters(), lr=self.params.lr),
            "ASGD": torch.optim.ASGD(self.parameters(), lr=self.params.lr)
        }
        optimizer = optim_dict[self.params.optimizer]
        ################################################
        optimizer = torch.optim.Adam(self.parameters(), lr=self.params.lr)

        ################################################
        scheduler_dict = {
            "ReduceLROnPlateau": ReduceLROnPlateau(
                optimizer,
                "min",
                factor=float(self.params.lr_factor),
                patience=float(self.params.lr_patience),
                min_lr=float(self.params.lr_min),
            ),
            "StepLR": StepLR(optimizer, step_size=1, gamma=self.params.lr_factor),
            "MultiStepLR": MultiStepLR(
                optimizer, milestones=[1, 2, 3], gamma=self.params.lr_factor
            ),
            "ExponentialLR": ExponentialLR(optimizer, gamma=self.params.lr_factor),
            "CosineAnnealingLR": CosineAnnealingLR(
                optimizer, T_max=10, eta_min=float(self.params.lr_min)
            ),
            "CyclicLR": CyclicLR(
                optimizer,
                base_lr=self.params.lr_min,
                max_lr=self.params.lr,
                step_size_up=10,
                cycle_momentum=False,
            ),
            "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts(
                optimizer, T_0=10, T_mult=1, eta_min=float(self.params.lr_min)
            ),
        }

        lr_scheduler = {
            "scheduler": scheduler_dict[self.params.lr_scheduler],
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
