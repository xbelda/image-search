import numpy as np
import torch
from typing import List, Dict


def in_batch_recall_at_1(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
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


def hit_rate(true_ids: np.ndarray, predicted_ids: np.ndarray, k: int | List[int]) -> Dict[int, float]:
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
