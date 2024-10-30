import os
from typing import Tuple, Dict, Any

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

from data_mlp import get_dataloader
from mlp_metrics import precision_test, cindex 
from Models.MLP.model import MLP_SODEN

def train_one_epoch(model: torch.nn.Module, train_dataloader: DataLoader, optimizer: Optimizer, epoch: int) -> float:
    """
    Train the model for one epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_dataloader (DataLoader): DataLoader for training data.
        optimizer (Optimizer): The optimizer for model parameters.
        epoch (int): The current epoch number.

    Returns:
        float: Average training loss for the epoch.
    """
    model.train()
    tr_loss = 0
    temp_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    counter = 0

    for step, batch in enumerate(train_dataloader):
        counter += 1
        inputs, label = batch
        logits, loss = model(inputs, label)
        loss.backward()
        temp_loss += loss.item()
        tr_loss += loss.item()
        nb_tr_examples += inputs['features'].size(0)
        nb_tr_steps += 1

        if step % 100 == 0:
            print(f"epoch: {epoch}\t| Cnt: {counter}\t| Loss: {temp_loss / 100}")
            temp_loss = 0

        optimizer.step()
        optimizer.zero_grad()

    return tr_loss / nb_tr_steps

def validate_one_epoch(model: torch.nn.Module, valid_dataloader: DataLoader, end_timepoint: int = 12, return_ph: bool = False) -> Tuple[float, float, float] or Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Validate the model for one epoch.

    Args:
        model (torch.nn.Module): The model to validate.
        valid_dataloader (DataLoader): DataLoader for validation data.
        end_timepoint (int, optional): The end timepoint for evaluation. Defaults to 12.
        return_ph (bool, optional): Whether to return partial hazards. Defaults to False.

    Returns:
        If return_ph is False:
            Tuple[float, float, float]: Precision, AUROC, and average loss.
        If return_ph is True:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Labels, hazard sequences, and times.
    """
    model.eval()
    y_label, y, hazard_seqs, y_time = [], [], [], []
    loss_temp = 0
    counter = 0

    for step, batch in enumerate(valid_dataloader):
        counter += 1
        inputs, label = batch
        with torch.no_grad():
            logits, loss = model(inputs, label)
        time2event = inputs['t']
        y_label.append(label)
        y.append(logits[0])
        hazard_seqs.append(logits[1])
        y_time.append(time2event)
        loss_temp += loss.item()

    y_label = torch.cat(y_label, dim=0)
    y = torch.cat(y, dim=0)
    y_time = torch.cat(y_time, dim=0)
    hazard_seqs = torch.cat(hazard_seqs, dim=0)
    
    tempprc, _, _ = precision_test(hazard_seqs[:, end_timepoint], y_label)
    tempauroc = cindex(hazard_seqs[:, end_timepoint], y_label, y_time)

    if not return_ph:
        return tempprc, tempauroc, loss_temp / counter
    return y_label, hazard_seqs, y_time

def main_training_loop(model: torch.nn.Module, train_filepath: str, test_filepath: str, model_path: str, optimizer: Optimizer, 
                       num_epochs: int, patience: int, scheduler: _LRScheduler, device: torch.device) -> Tuple[float, float, float, torch.nn.Module]:
    """
    Main training loop for the model.

    Args:
        model (torch.nn.Module): The model to train.
        train_filepath (str): Path to the training data file.
        test_filepath (str): Path to the testing data file.
        model_path (str): Path to save the model to
        optimizer (Optimizer): The optimizer for model parameters.
        num_epochs (int): Maximum number of epochs to train.
        patience (int): Number of epochs to wait for improvement before early stopping.
        scheduler (_LRScheduler): Learning rate scheduler.
        device (torch.device): Device to run the model on.

    Returns:
        Tuple[float, float, float, torch.nn.Module]: Final precision, AUROC, testing loss, and the trained model.
    """
    best_test_loss = float('inf')
    patience_counter = 0

    train_dataloader, _ = get_dataloader(train_filepath, batch_size=512, std=1)
    test_dataloader, _ = get_dataloader(test_filepath, batch_size=256, std=1)

    for epoch in range(1, num_epochs+1):
        train_loss = train_one_epoch(model, train_dataloader, optimizer, epoch)
        tempprc, tempauroc, test_loss = validate_one_epoch(model, test_dataloader)

        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Testing Loss: {test_loss:.4f}, TempPRC: {tempprc:.4f}, TempAUROC: {tempauroc:.4f}")

        scheduler.step(test_loss)
        if test_loss < best_test_loss:
            best_test_loss = test_loss
            patience_counter = 0
            torch.save(model.state_dict(), f'{model_path}/{model.suffix}_model.pth')
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping after {epoch} epochs due to no improvement in testing loss.")
            break

    return tempprc, tempauroc, test_loss, model

def load_model(config: Dict[str, any], model_path: str) -> MLP_SODEN:
    """
    Load the MLP_SODEN model with the given configuration and weights.

    Args:
        config (Dict[str, any]): Model configuration dictionary.
        model_path (str): Path to the saved model weights.

    Returns:
        MLP_SODEN: Loaded model.
    """
    model = MLP_SODEN(config)
    model.load_state_dict(torch.load(f"{model_path}/{config['features']}_model.pth"))
    return model

def get_validation_results(model: MLP_SODEN, data_folder: str, suffix: str, batch_size: int = 256, std: float = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Get validation data from the model.

    Args:
        model (MLP_SODEN): The loaded model.
        data_folder (str): Path to the data folder.
        suffix (str): Data file suffix.
        batch_size (int, optional): Batch size for dataloader. Defaults to 256.
        std (float, optional): Standard deviation for dataloader. Defaults to 1.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Event, partial hazard, and time tensors.
    """
    path = os.path.join(data_folder, f'valid_{suffix}.pt')
    valid_dataloader, _ = get_dataloader(path, batch_size=batch_size, std=std)
    return validate_one_epoch(model, valid_dataloader, return_ph=True)