from pathlib import Path
from typing import Tuple

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel

from image_search.data import ConversionsDataset, ImagesDataset
from image_search.model import LightningImageSearchSigLIP

# CONSTANTS
IMAGES_FOLDER = Path("./data/unsplash-research-dataset-lite-latest/photos/")
BASE_MODEL = "google/siglip-base-patch16-224"
BATCH_SIZE = 128
NUM_WORKERS = 16
SEED = 42
LR = 1e-4


def load_and_preprocess_data():
    conversions = pd.read_csv(
        "./data/unsplash-research-dataset-lite-latest/conversions.tsv000",
        sep="\t",
        header=0,
    )
    conversions["converted_at"] = pd.to_datetime(
        conversions["converted_at"], format="ISO8601"
    )
    # Sort by conversion datetime
    conversions = conversions.sort_values("converted_at", ignore_index=True)
    return conversions


def temporal_train_test_split(
    conversions: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Temporally splits the dataset in Train/Test
    This is one of the best approaches, since it allows us to measure more how the model would work under a more
    "realistic" scenario. That is, training a model on previous data and seeing how it evolves in the future.
    """
    num_examples = len(conversions)
    conversions_train = conversions[: int(num_examples * 0.8)]
    conversions_val = conversions[int(num_examples * 0.8) :]
    return conversions_train, conversions_val


def main():
    torch.set_float32_matmul_precision("medium")  # Improves speed using tensor cores
    pl.seed_everything(SEED)

    # conversions = load_and_preprocess_data()
    conversions = pd.read_parquet("./data/clean/conversions.parquet")
    conversions_train, conversions_val = temporal_train_test_split(conversions)

    # Dataset
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    image_dataset = ImagesDataset(image_folder=IMAGES_FOLDER, processor=processor)
    dataset_train = ConversionsDataset(
        data=conversions_train, image_dataset=image_dataset, processor=processor
    )
    dataset_val = ConversionsDataset(
        data=conversions_val, image_dataset=image_dataset, processor=processor
    )

    dataloader_train = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
    )

    # Train
    model = AutoModel.from_pretrained(BASE_MODEL)
    lightning_model = LightningImageSearchSigLIP(model=model, lr=LR)

    logger = pl.loggers.MLFlowLogger(experiment_name="ImageSearch")

    trainer = pl.Trainer(
        logger=logger, max_epochs=2, precision="bf16-mixed", log_every_n_steps=20
    )
    trainer.fit(lightning_model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()
