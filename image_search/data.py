from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from PIL import Image


class ConversionsDataset(torch.utils.data.Dataset):
    def __init__(self, data: pd.DataFrame, image_folder: Path):
        self.dataset = Dataset.from_pandas(data[["keyword", "photo_id"]])
        self.images_folder = image_folder

    def get_image_path(self, photo_id):
        return (self.images_folder / photo_id).with_suffix(".jpg")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]

        keyword = example["keyword"]

        image_path = self.get_image_path(example["photo_id"])
        image = Image.open(image_path).convert(
            'RGB')  # Note: Conversion to RGB format is crucial to avoid issues with JPG/PNG format

        return {"keyword": keyword, "image": image}


class SigLIPCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, examples):
        keywords = [example["keyword"] for example in examples]
        images = [example["image"] for example in examples]

        # Important: we pass `padding=max_length` since the model was trained with this.
        # Changing it might lead to unexpected behavior
        inputs = self.processor(text=keywords, images=images, padding="max_length", truncation=True,
                                return_tensors="pt")
        return inputs
