import pytorch_lightning as pl
import torch
import torch.nn.functional as F

from image_search.metrics import in_batch_recall_at_1
from transformers import SiglipModel


class QueryModel(torch.nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.text_model = model.text_model

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        text_outputs = self.text_model(input_ids=input_ids, return_dict=True)

        query_embeddings = text_outputs.pooler_output

        # normalized features
        query_embeddings = F.normalize(query_embeddings)

        return query_embeddings


class ImageModel(torch.nn.Module):
    def __init__(self, model: SiglipModel):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_output = self.vision_model(pixel_values=pixel_values, return_dict=True)

        image_embeddings = image_output.pooler_output

        # normalized features
        image_embeddings = F.normalize(image_embeddings)

        return image_embeddings


class LightningImageSearchSigLIP(pl.LightningModule):
    def __init__(self, model: SiglipModel, lr: float):
        super().__init__()

        self.query_model = QueryModel(model)
        self.image_model = ImageModel(model)

        self.logit_scale = model.logit_scale
        self.logit_bias = model.logit_bias

        self.lr = lr

    def _base_step(self, input_ids, pixel_values):
        query_embeddings = self.query_model(input_ids)
        image_embeddings = self.image_model(pixel_values)

        # cosine similarity as logits
        logits_per_query = torch.matmul(query_embeddings,
                                        image_embeddings.t()) * self.logit_scale.exp() + self.logit_bias
        logits_per_query = logits_per_query.t()

        batch_size = logits_per_query.shape[0]
        targets = (2 * torch.eye(batch_size) - torch.ones(batch_size))
        targets = targets.to(logits_per_query.device)

        loss = siglip_loss(logits=logits_per_query, targets=targets)
        recall = in_batch_recall_at_1(logits_per_query, targets)
        return loss, recall

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss, recall = self._base_step(input_ids, pixel_values)
        self.log("Loss/Train", loss)
        self.log("InBatchRecallAt1/Train", recall)
        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        pixel_values = batch["pixel_values"]

        loss, recall = self._base_step(input_ids, pixel_values)
        self.log("Loss/Val", loss)
        self.log("InBatchRecallAt1/Val", recall)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


def siglip_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(logits * targets).sigmoid().log().sum(dim=1).mean()
