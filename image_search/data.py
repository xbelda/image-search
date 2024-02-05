from pathlib import Path

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

        self.image_paths = sorted(image_folder.glob('*.jpg'))
        self.names = [path.stem for path in self.image_paths]

        self.ids_name = dict(enumerate(self.names))
        self.name_ids = {name: idx for idx, name in self.ids_name.items()}

    def id_to_name(self, idx: int) -> str:
        return self.ids_name[idx]

    def name_to_id(self, name: str) -> int:
        return self.name_ids[name]

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx):
        name = self.names[idx]
        image_path = self.image_paths[idx]

        idx = self.name_to_id(name)

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor.image_processor(image, return_tensors="pt")

        return {
            "photo_id": idx,
            "pixel_values": inputs.pixel_values.squeeze()
        }


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
