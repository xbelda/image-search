from functools import cache
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image
from transformers import SiglipProcessor
from tqdm import tqdm


class ImagesDataset(torch.utils.data.Dataset):
    """
    Extract all available images from a `image_folder` directory.
    Returns the name and the loaded image.
    """

    def __init__(
        self,
        processor: SiglipProcessor,
        image_folder: Path,
        keywords: pd.DataFrame,
    ):
        self.processor = processor
        self.images_folder = image_folder

        self.names = sorted(keywords["photo_id"].unique())
        self.photo_keyword_ids = (
            keywords.groupby("photo_id")["keyword_id"].apply(list).to_dict()
        )

    def id_to_name(self, idx: int) -> str:
        return self.names[idx]

    def name_to_id(self, name: str) -> int:
        return self.names.index(name)

    def __len__(self) -> int:
        return len(self.names)

    def pre_cache_images(self):
        print("Pre-caching images...")
        for _ in tqdm(self):
            pass

    @cache
    def _load_image(self, name: str) -> torch.Tensor:
        image_path = self.images_folder / f"{name}.jpg"
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor.image_processor(image, return_tensors="pt")
        return inputs

    def __getitem__(self, idx: int) -> Dict[str, int | torch.Tensor]:
        name = self.names[idx]

        inputs = self._load_image(name)
        pixel_values = inputs.pixel_values.squeeze()

        tags = self.photo_keyword_ids[name]

        return {"id": idx, "pixel_values": pixel_values, "tags": tags}


class ConversionsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        image_dataset: ImagesDataset,
        processor: SiglipProcessor,
    ):
        self.dataset = Dataset.from_pandas(
            data[["keyword", "photo_id", "conversion_country"]]
        )
        self.image_dataset = image_dataset
        self.processor = processor

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        keyword = example["keyword"]
        photo_id = example["photo_id"]
        country = example["conversion_country"]

        # The simplest way to add the country is to add it to the textual model
        # It is also possible to add it as an additional embedding
        text = f"{country}: {keyword}"

        image_id = self.image_dataset.name_to_id(photo_id)
        image_data = self.image_dataset[image_id]

        # TODO: Add image augmentation

        ids = image_data["id"]
        pixel_values = image_data["pixel_values"].squeeze()
        tags = image_data["tags"]

        text_inputs = self.processor.tokenizer(
            text=text, padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids = text_inputs["input_ids"].squeeze()

        return {
            "ids": ids,
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "tags": tags,
        }


def collate_tags(examples: List[Dict]) -> Dict[str, torch.Tensor]:
    tags = [torch.tensor(example.pop("tags")) for example in examples]

    batch = torch.utils.data.default_collate(examples)
    batch["tags"] = torch.nn.utils.rnn.pad_sequence(
        tags, batch_first=True, padding_value=0
    )

    return batch
