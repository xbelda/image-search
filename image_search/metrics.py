import torch


def in_batch_recall_at_1(predictions, targets):
    predicted_ids = predictions.argmax(dim=1)
    expected_ids = targets.argmax(dim=1)

    return (predicted_ids == expected_ids).to(torch.float).mean()
