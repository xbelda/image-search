import pytorch_lightning as pl
import torch

from image_search.metrics import in_batch_recall_at_1
from transformers import SiglipModel


class LightningImageSearchSigLIP(pl.LightningModule):
    def __init__(self, model: SiglipModel, lr: float):
        super().__init__()
        self.model = model
        self.lr = lr

    def _basic_step(self, input_ids, pixel_values):
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image

        batch_size = logits_per_image.shape[0]
        targets = (2 * torch.eye(batch_size) - torch.ones(batch_size))
        targets = targets.to(logits_per_image.device)

        loss = siglip_loss(logits=logits_per_image, targets=targets)
        recall = in_batch_recall_at_1(logits_per_image, targets)
        return loss, recall

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss, recall = self._basic_step(input_ids, pixel_values)
        self.log("Loss/Train", loss)
        self.log("InBatchRecallAt1/Train", recall)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss, recall = self._basic_step(input_ids, pixel_values)
        self.log("Loss/Val", loss)
        self.log("InBatchRecallAt1/Val", recall)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def siglip_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(logits * targets).sigmoid().log().sum(dim=1).mean()
