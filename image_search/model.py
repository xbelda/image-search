import pytorch_lightning as pl
import torch
from transformers import Adafactor


class LightningImageSearchSigLIP(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _basic_step(self, input_ids, pixel_values):
        outputs = self.model(input_ids=input_ids, pixel_values=pixel_values)
        logits_per_image = outputs.logits_per_image
        # logits_per_text = outputs.logits_per_text

        batch_size = logits_per_image.shape[0]
        target = (2 * torch.eye(batch_size) - torch.ones(batch_size))
        target = target.to(logits_per_image.device)

        loss = -(logits_per_image * target).sigmoid().log().sum(dim=1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss = self._basic_step(input_ids, pixel_values)
        self.log("Loss/Train", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss = self._basic_step(input_ids, pixel_values)
        self.log("Loss/Val", loss)
        return loss

    def configure_optimizers(self):
        # optimizer = Adafactor(self.parameters())
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer


class SigLIPLoss(torch.nn.Module):
    """WARNING: WIP"""
    def __init__(self, logit_scale: torch.tensor, logit_bias: torch.tensor):
        super().__init__()
        # Ideally, these parameters should be initialized to the same
        self.logit_scale = torch.nn.Parameter(logit_scale)
        self.logit_bias = torch.nn.Parameter(logit_bias)

    def forward(self, img_emb, txt_emb, target):
        raise NotImplementedError
        # batch_size = img_emb.shape[0]
        #
        # logits = img_emb @ txt_emb.T * self.logit_scale.exp() + self.logit_bias
        # loss = - (1 / batch_size) * (logits * target).sigmoid().log()
        #
        # return loss.sum(0).mean()
