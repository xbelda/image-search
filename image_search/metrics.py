from typing import List, Dict

import numpy as np
import torch


def in_batch_recall_at_1(
    predictions: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    Computes the recall at 1 metric for a batch of predictions.
    Args:
        predictions:
        targets:

    Returns:

    """
    predicted_ids = predictions.argmax(dim=1)
    expected_ids = targets.argmax(dim=1)

    return (predicted_ids == expected_ids).to(torch.float).mean()


def hit_rate(
    true_ids: np.ndarray, predicted_ids: np.ndarray, k: int | List[int]
) -> Dict[int, float]:
    """
    Computes the hit rate of a batch of predictions for different values of k.

    Args:
        true_ids: Array containing the true labels for the batch.
        predicted_ids: Array containing the predicted labels for the batch.
        k: Array of different cutoff values for which to compute the hit rate.
          If an integer is provided, computes only that value.

    Returns:
        Dictionary containing the hit rate at different values of k.
    """

    if isinstance(k, int):
        k = [k]

    score = {}
    for current_k in k:
        predicted_ids_k = predicted_ids[:, :current_k]
        hit_k = (predicted_ids_k == true_ids.T).any(axis=1).mean()

        score[current_k] = hit_k
    return score


def mean_average_precision(true_ids: np.ndarray, predicted_ids: np.ndarray) -> float:
    """
    Computes the average precision of a batch of predictions.
    Args:
        true_ids: Array containing the true labels for the batch.
        predicted_ids: Array containing the predicted labels for the batch.

    Returns:
        Average_precision.

    """

    matches = predicted_ids == true_ids.T

    # A is relevance
    precision = matches.cumsum(axis=1) / (1 + np.arange(matches.shape[1]))
    masked_precision = precision * matches

    num_correct = matches.sum(axis=1) + 1e-9  # Add epsilon to avoid division by zero

    _average_precision = masked_precision.sum(axis=1) / num_correct
    return _average_precision.mean()


def mean_average_precision_at_k(
    true_ids: np.ndarray, predicted_ids: np.ndarray, k: int | List[int]
) -> Dict[int, float]:
    """
    Computes the average precision of a batch of predictions for different values of k.
    Args:
        true_ids: Array containing the true labels for the batch.
        predicted_ids: Array containing the predicted labels for the batch.
        k: Array of different cutoff values for which to compute the hit rate.
          If an integer is provided, computes only that value.

    Returns:
        Dictionary containing the average_precision at different values of k.
    """

    if isinstance(k, int):
        k = [k]

    score = {}
    for current_k in k:
        predicted_ids_k = predicted_ids[:, :current_k]
        map_at_k = mean_average_precision(true_ids, predicted_ids_k)
        score[current_k] = map_at_k
    return score
