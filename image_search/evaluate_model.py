import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from image_search.data import ImagesDataset, ConversionsDataset, collate_tags
from image_search.metrics import hit_rate, mean_average_precision_at_k
from image_search.model import LightningImageSearchSigLIP, ImageModel, QueryModel
from image_search.preprocessing import KeywordProcessor, temporal_train_test_split, load_and_preprocess_data

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='example.log',
    filemode='a'
)

# CONFIG
IMAGES_FOLDER = Path("./data/unsplash-research-dataset-lite-latest/photos/")
KEYWORDS_PATH = Path("./data/unsplash-research-dataset-lite-latest/keywords.tsv000")

BASE_MODEL = "google/siglip-base-patch16-224"

BATCH_SIZE = 2048
NUM_WORKERS = 4
SEED = 42

CHECKPOINT_PATHS = {
    "VANILLA": None,
    "FINE-TUNED": "./mlruns/168016125050379525/b4956e72f44f4df48b395cf338545014/checkpoints/epoch=0-step=60598.ckpt",
    "COUNTRY&EMBEDDINGS": "./mlruns/168016125050379525/da08e024170a4fac9fd77335575e865f/checkpoints/epoch=0-step=60598.ckpt",
}

MODEL_NAME = "COUNTRY&EMBEDDINGS"
CHECKPOINT_PATH = CHECKPOINT_PATHS[MODEL_NAME]
OUTPUT_PATH = Path("./results/") / f"{MODEL_NAME}.csv"


@dataclass
class AcceleratorConfig:
    device: str
    dtype: torch.dtype


def load_models(accelerator):
    """Loads the model, splits it into the image and query models, and puts them into eval mode."""
    model = AutoModel.from_pretrained(BASE_MODEL).to(accelerator.device)

    if CHECKPOINT_PATH is not None:
        lightning_model = LightningImageSearchSigLIP.load_from_checkpoint(CHECKPOINT_PATH, model=model, lr=1e-4)
    else:
        # In case we want to load the vanilla model directly
        lightning_model = LightningImageSearchSigLIP(model=model, lr=1e-4)

    lightning_model = lightning_model.to(device=accelerator.device, dtype=accelerator.dtype)
    image_model = lightning_model.image_model
    query_model = lightning_model.query_model

    # Enable eval mode
    image_model = image_model.eval()
    query_model = query_model.eval()
    return image_model, query_model


def get_image_dataloader(df_keywords: pd.DataFrame, processor) -> DataLoader:
    image_dataset = ImagesDataset(processor=processor, image_folder=IMAGES_FOLDER, keywords=df_keywords)
    image_dataset.pre_cache_images()
    image_dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                                                   collate_fn=collate_tags)
    return image_dataloader


def get_index(image_dataloader: DataLoader,
              image_model: ImageModel,
              accelerator: AcceleratorConfig) -> faiss.IndexFlatL2:
    """Generate Index for Images using Faiss.

    This implementation the [Pinecone guide](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/) for faiss.

    Note: Since the dataset is relatively small, a flat Faiss index is used for simplicity and efficiency.
    In a real scenario with a larger dataset, optimization for faster retrieval may be necessary.

    Args:
        image_dataloader (torch.utils.data.DataLoader): The data loader containing the images.
        image_model (torch.nn.Module): The model used to generate embeddings for images.
        accelerator (accelerate.Accelerator): The accelerator device to use for computations.

    Returns:
        faiss.IndexFlatL2: A Faiss index containing the embeddings of the images.

"""
    embedding_dim = image_model.vision_model.config.hidden_size
    index = faiss.IndexFlatL2(embedding_dim)
    for batch in tqdm(image_dataloader):
        pixel_values = batch["pixel_values"].to(device=accelerator.device, dtype=accelerator.dtype)
        tags = batch["tags"].to(device=accelerator.device)

        image_embeddings = image_model(pixel_values=pixel_values, tags=tags)
        image_embeddings = image_embeddings.to(device="cpu", dtype=torch.float32).detach().numpy()

        index.add(image_embeddings)
    return index


def get_conversions_dataloader(conversions_val, df_keywords, processor):
    image_dataset = ImagesDataset(processor=processor, image_folder=IMAGES_FOLDER, keywords=df_keywords)
    dataset_val = ConversionsDataset(data=conversions_val, image_dataset=image_dataset, processor=processor)
    dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, collate_fn=collate_tags)
    return dataloader_val


def generate_query_recommendations(query_model: QueryModel,
                                   dataloader_val: DataLoader,
                                   index: faiss.IndexFlatL2,
                                   accelerator: AcceleratorConfig) -> [np.ndarray, np.ndarray]:
    true_ids = []
    predicted_ids = []
    for batch in tqdm(dataloader_val):
        input_ids = batch["input_ids"].to(accelerator.device)
        ids = batch["ids"].unsqueeze(dim=0).numpy()

        query_embedding = query_model(input_ids)
        query_embedding = query_embedding.to(device="cpu", dtype=torch.float32).detach().numpy()

        distances, indices = index.search(query_embedding, k=25)

        true_ids.append(ids)
        predicted_ids.append(indices)
    true_ids = np.concatenate(true_ids, axis=1)
    predicted_ids = np.concatenate(predicted_ids, axis=0)
    return predicted_ids, true_ids


def main():
    logging.info("Disabling gradients for inference")
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        accelerator = AcceleratorConfig(device="cuda", dtype=torch.bfloat16)
    else:
        accelerator = AcceleratorConfig(device="cpu", dtype=torch.float32)

    logging.info("Loading model and processor")
    processor = AutoProcessor.from_pretrained(BASE_MODEL)
    image_model, query_model = load_models(accelerator)

    logging.info("Loading keywords")
    df_keywords = pd.read_csv(KEYWORDS_PATH, sep='\t')
    keyword_processor = KeywordProcessor()
    df_keywords = keyword_processor.process(df_keywords)

    logging.info("Generating images dataset")
    image_dataloader = get_image_dataloader(df_keywords, processor)

    logging.info("Indexing images")
    index = get_index(image_dataloader, image_model, accelerator)
    # To avoid re-computing all indices, we will save the current index
    faiss.write_index(index, f"indices/{MODEL_NAME}.index")

    logging.info("Loading conversions")
    conversions = load_and_preprocess_data()
    _, conversions_val = temporal_train_test_split(conversions)
    dataloader_val = get_conversions_dataloader(conversions_val, df_keywords, processor)

    logging.info("Generating query recommendations")
    predicted_ids, true_ids = generate_query_recommendations(query_model, dataloader_val, index, accelerator)

    logging.info("Evaluating metrics")
    k = [1, 5, 10, 25]

    hit_rate_score = hit_rate(true_ids, predicted_ids, k=k)
    map_score = mean_average_precision_at_k(true_ids=true_ids, predicted_ids=predicted_ids, k=k)

    scores = pd.DataFrame({
        "hit_rate": hit_rate_score,
        "mAP": map_score,
    })
    scores.index.name = "k"
    print(scores)

    scores.to_csv(OUTPUT_PATH)


if __name__ == "__main__":
    main()
