import json
from src.metrics import rawconfinterval
from sklearn.metrics import average_precision_score, roc_auc_score

from typing import Tuple, List
import numpy as np
import torch

def load_json_config(filepath):
    """
    Load JSON configuration file.

    Args:
        filepath (str): Path to the JSON file.

    Returns:
        dict: Loaded JSON data.
    """
    with open(filepath, 'r') as file:
        config =  json.load(file)

    field_mapping = config['field_mapping']
    for key in ['selected_columns', 'maggic_columns', 'maggic_plus_columns', 'int_columns', 'float_columns']:
        config[key] = map_fields(config[key], field_mapping)

    return config

def map_fields(fields, field_mapping):
    """
    Map field names using the provided mapping.

    Args:
        fields (list): List of field names to map.
        field_mapping (dict): Dictionary containing field name mappings.

    Returns:
        list: Mapped field names.
    """
    return [field_mapping[field] for field in fields]

    
def convpretty(arr: List[Tuple[float, float]], num: int = 3) -> List[str]:
    """
    Convert array of means and confidence intervals to pretty string format.

    Args:
        arr (List[Tuple[float, float]]): List of (mean, confidence_interval) tuples.
        num (int): Number of decimal places to round to.

    Returns:
        List[str]: List of formatted strings.
    """
    def truncate(value: float, digits: int) -> str:
        return f"{value:.{digits}f}"

    return [f"{truncate(x[0], num)} ({truncate(x[0] - x[1], num)}, {truncate(x[0] + x[1], num)})" for x in arr]

def save_outputs(suffix: str, t: torch.Tensor, e: torch.Tensor, ph: torch.Tensor, timepoints: list):
    """
    Save model outputs for different timepoints.

    Args:
        suffix (str): File suffix for saving.
        t (torch.Tensor): Time tensor.
        e (torch.Tensor): Event tensor.
        ph (torch.Tensor): Partial hazard tensor.
        timepoints (list): List of timepoints to save.
    """
    for timepoint in timepoints:
        np.savez(
            f'./../outputs/{suffix}_{timepoint}.npz',
            time=t.flatten().numpy(),
            event=e.flatten().numpy(),
            pred=ph.cpu().numpy()[:, timepoint],
            partial_hazard=ph.cpu().numpy()
        )



    