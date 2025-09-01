import logging
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModel

from image_search.data import ConversionsDataset, ImagesDataset, collate_tags
from image_search.model import LightningImageSearchSigLIP
from image_search.preprocessing import KeywordProcessor, temporal_train_test_split

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename="example.log",
    filemode="a",
)

# CONSTANTS
IMAGES_FOLDER = Path("./data/unsplash-research-dataset-lite-latest/photos/")
BASE_MODEL = "google/siglip-base-patch16-224"
BATCH_SIZE = 128
NUM_WORKERS = 4
SEED = 42
LR = 1e-4
NUM_EPOCHS = 1
RUN_NAME = "COUNTRY+EMBEDDINGS"


def main():
    torch.set_float32_matmul_precision("medium")  # Improves speed using tensor cores
    pl.seed_everything(SEED)

    logging.info("Loading data")
    # conversions = load_and_preprocess_data()
    conversions = pd.read_parquet("./data/clean/conversions.parquet")
    conversions_train, conversions_val = temporal_train_test_split(conversions)

    df_keywords = pd.read_csv(
        "./data/unsplash-research-dataset-lite-latest/keywords.tsv000", sep="\t"
    )

    keyword_processor = KeywordProcessor()
    df_keywords = keyword_processor.process(df_keywords)

    print("Loading processor")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)

    print("Generating datasets")
    image_dataset = ImagesDataset(
        processor=processor, image_folder=IMAGES_FOLDER, keywords=df_keywords
    )
    image_dataset.pre_cache_images()

    dataset_train = ConversionsDataset(
        data=conversions_train, image_dataset=image_dataset, processor=processor
    )
    dataset_val = ConversionsDataset(
        data=conversions_val, image_dataset=image_dataset, processor=processor
    )

    dataloader_train = DataLoader(
        dataset_train,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_tags,
    )
    dataloader_val = DataLoader(
        dataset_val,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        collate_fn=collate_tags,
    )

    print("Setting up model")
    model = AutoModel.from_pretrained(BASE_MODEL)
    lightning_model = LightningImageSearchSigLIP(model=model, lr=LR)

    logger = pl.loggers.MLFlowLogger(experiment_name="ImageSearch", run_name=RUN_NAME)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=NUM_EPOCHS,
        precision="bf16-mixed",
        log_every_n_steps=20,
    )
    trainer.fit(lightning_model, dataloader_train, dataloader_val)


if __name__ == "__main__":
    main()
