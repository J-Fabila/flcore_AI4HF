import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning import LightningDataModule

class MMsDataSet(LightningDataModule):
    pass

class LightningWrapperData(LightningDataModule):
    def __init__(self, params):
        super(LightningWrapperData, self).__init__()
        self.data = torch.load(params.dataset_root)
        self.params = params

        assert ( self.params.train_size + self.params.val_size + self.params.test_size ) <= 1.0 , "Sum of train + validation + test is larger than 1.0"

        if torch.cuda.is_available() and self.params.device == 'cuda':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device("cpu")
        
    def setup(self, stage):
        self.len = len(self.data)
        indices = np.arange(self.len)
        np.random.shuffle(indices)

    #if stage == "train":
        train_size=int(self.params.train_size * self.len)
        self.ind_train = indices[:train_size]
        self.train_size = len(self.ind_train)
        self.train = Subset(self.data, self.ind_train)
        """for config in range(train_size):
            for key, value in self.train[config].items():
                if key in train:       
                    self.train[config][key] = value.to(self.device)
        """       
    #if stage == "val":
        val_size=int(self.params.val_size * self.len)
        self.ind_val = indices[train_size:val_size+train_size]
        self.val_size = len(self.ind_val)
        self.val = Subset(self.data, self.ind_val)        

        """for config in range(val_size):
            for key, value in self.val[config].items():
                if key in train:       
                    self.val[config][key] = value.to(self.device)
        """
        test_size=int(self.params.test_size * self.len )
        self.ind_test = indices[train_size+val_size:]
        self.test_size = len(self.ind_test)
        self.test = Subset(self.data, self.ind_test)


    def train_dataloader(self):
        return self._get_dataloader(self.train, "train")

    def val_dataloader(self):
        return self._get_dataloader(self.val, "val")

    def test_dataloader(self):

        return self._get_dataloader(self.test, "test")

    def _get_dataloader(self, dataset, stage, store_dataloader=True):
        if stage == "train":
            shuffle_ = True
        else:
            shuffle_ = False

        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.params.batch_size,
            num_workers=self.params.num_workers,
            pin_memory=True,
            shuffle=shuffle_,
        )
        return data_loader
