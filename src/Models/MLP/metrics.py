from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score
import numpy as np
import torch.nn as nn
from typing import Tuple, List
import torch


def cindex(partial_hazard: np.ndarray, event: np.ndarray, time: np.ndarray) -> float:
    """
    Calculate the concordance index.

    Args:
        partial_hazard (np.ndarray): Partial hazard predictions.
        event (np.ndarray): Event indicators.
        time (np.ndarray): Survival times.

    Returns:
        float: Concordance index.
    """
    return concordance_index(time.cpu(), -1 * partial_hazard.cpu(), event.cpu())

def roc_auc(logits: torch.Tensor, label: torch.Tensor, sig: bool = True) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculate the ROC AUC score.

    Args:
        logits (torch.Tensor): Model output logits.
        label (torch.Tensor): True labels.
        sig (bool): Whether to apply sigmoid to logits.

    Returns:
        Tuple[float, torch.Tensor, torch.Tensor]: ROC AUC score, processed output, and labels.
    """
    sigm = nn.Sigmoid()
    output = sigm(logits) if sig else logits
    label, output = label.cpu(), output.detach().cpu()
    roc_score = roc_auc_score(label.numpy(), output.numpy())
    return roc_score, output, label

def precision(logits: torch.Tensor, label: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculate the average precision score.

    Args:
        logits (torch.Tensor): Model output logits.
        label (torch.Tensor): True labels.

    Returns:
        Tuple[float, torch.Tensor, torch.Tensor]: Average precision score, processed output, and labels.
    """
    sig = nn.Sigmoid()
    output = sig(logits)
    label, output = label.cpu(), output.detach().cpu()
    avg_precision = average_precision_score(label.numpy(), output.numpy())
    return avg_precision, output, label

def precision_test(logits: torch.Tensor, label: torch.Tensor) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """
    Calculate the average precision score for test data.

    Args:
        logits (torch.Tensor): Model output logits.
        label (torch.Tensor): True labels.

    Returns:
        Tuple[float, torch.Tensor, torch.Tensor]: Average precision score, processed output, and labels.
    """
    sig = nn.Sigmoid()
    output = sig(logits)
    avg_precision = average_precision_score(label.cpu().numpy(), output.cpu().numpy())
    return avg_precision, output, label

def confinterval(metricval: float, n1: int, n2: int) -> Tuple[float, float]:
    """
    Calculate the confidence interval for a metric.

    Args:
        metricval (float): Metric value.
        n1 (int): Number of positive samples.
        n2 (int): Number of negative samples.

    Returns:
        Tuple[float, float]: Lower and upper bounds of the confidence interval.
    """
    q0 = metricval * (1 - metricval)
    q1 = (metricval / (2 - metricval)) - (metricval ** 2)
    q2 = ((2 * metricval) / (1 + metricval)) - (metricval ** 2)
    
    se = np.sqrt((q0 + q1 * (n1 - 1) + q2 * (n2 - 1)) / (n1 * n2))
    lower_bound = metricval - (1.96 * se)
    upper_bound = metricval + (1.96 * se)
    
    return lower_bound, upper_bound

def rawconfinterval(metricval: float, n1: int, n2: int) -> float:
    """
    Calculate the raw confidence interval for a metric.

    Args:
        metricval (float): Metric value.
        n1 (int): Number of positive samples.
        n2 (int): Number of negative samples.

    Returns:
        float: Raw confidence interval.
    """
    q0 = metricval * (1 - metricval)
    q1 = (metricval / (2 - metricval)) - (metricval ** 2)
    q2 = ((2 * metricval) / (1 + metricval)) - (metricval ** 2)
    
    se = np.sqrt((q0 + q1 * (n1 - 1) + q2 * (n2 - 1)) / (n1 * n2))
    confidence_interval = 1.96 * se
    
    return confidence_interval

def format_conf_interval(mean: float, n_events: int) -> str:
    """
    Format the confidence interval as a string.

    Args:
        mean (float): Mean value.
        n_events (int): Number of events.

    Returns:
        str: Formatted confidence interval string.
    """
    return "; ".join([f"{x:.3f}" for x in confinterval(mean, n_events, 1 - n_events)])


