import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.callbacks import EarlyStopping

def train(modelo, params, dataset):

    #print(" train  ::  inicia")
    
    ## CSV Logger
    csv_logger = CSVLogger(params.log_path, name="", version="",prefix="")

    _logger=[csv_logger]
    ## Add WandB as option
    if params.wandb_track:
        wandb_logger=WandbLogger(project=params.wandb_project,name=params.wandb_run_name,save_dir=params.log_path)
        _logger.append(wandb_logger)
        
    ## Add TB also as an option
    model_checkpoint = ModelCheckpoint(
        dirpath=params.log_path,
        monitor="val_loss",
        save_top_k=params.save_top_k,
        every_n_epochs=params.every_n_epochs,
        filename="{epoch}-{val_loss:.4f}",
    )
    lr_monitor = LearningRateMonitor()
    rich_progress_bar = RichProgressBar()

    # We can add also early stopping
    early_stopping = EarlyStopping("val_loss", patience=params.early_stopping_patience)

    trainer = pl.Trainer(
        max_epochs=params.epochs,
        num_nodes=params.n_gpu_nodes,
        default_root_dir=params.log_path,
        callbacks=[model_checkpoint ,lr_monitor,rich_progress_bar],
        logger=_logger,
        gradient_clip_val=params.clip,
        log_every_n_steps = 1
    )
    trainer.fit(modelo, dataset)
    
    if params.federated:
        #print(" train  ::  terminaa")
        pass
    else:
        print("=================== TESTING ")
        trainer.test(dataloaders=dataset)
