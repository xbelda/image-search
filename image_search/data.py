from functools import cache
from pathlib import Path
from typing import Dict

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image

from transformers import SiglipProcessor


class ImagesDataset(torch.utils.data.Dataset):
    """
    Extract all available images from a `image_folder` directory.
    Returns the name and the loaded image.
    """

    def __init__(self, image_folder: Path, processor: SiglipProcessor):
        self.images_folder = image_folder
        self.processor = processor

        self.image_paths = sorted(
            image_folder.glob("*.jpg")
        )  # Sort paths for reproducibility (TODO: refactor this)
        self.names = [path.stem for path in self.image_paths]

        self.ids_name = dict(enumerate(self.names))
        self.name_ids = {name: idx for idx, name in self.ids_name.items()}

    def id_to_name(self, idx: int) -> str:
        return self.ids_name[idx]

    def name_to_id(self, name: str) -> int:
        return self.name_ids[name]

    def __len__(self) -> int:
        return len(self.image_paths)

    @cache
    def _load_image(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor.image_processor(image, return_tensors="pt")
        return inputs

    def __getitem__(self, idx: int) -> Dict[str, int | torch.Tensor]:
        image_path = self.image_paths[idx]

        inputs = self._load_image(image_path)

        return {"id": idx, "pixel_values": inputs.pixel_values.squeeze()}


class ConversionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        image_dataset: ImagesDataset,
        processor: SiglipProcessor,
    ):
        self.dataset = Dataset.from_pandas(data[["keyword", "photo_id"]])
        self.image_dataset = image_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        keyword = example["keyword"]
        photo_id = example["photo_id"]

        image_id = self.image_dataset.name_to_id(photo_id)
        image_data = self.image_dataset[image_id]

        ids = image_data["id"]
        pixel_values = image_data["pixel_values"].squeeze()

        text_inputs = self.processor.tokenizer(
            text=keyword, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze()

        return {"ids": ids, "input_ids": input_ids, "pixel_values": pixel_values}
