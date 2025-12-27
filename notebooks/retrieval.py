# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
# cd ..

# %%
from typing import Tuple
import numpy as np
import pandas as pd
from pathlib import Path
import faiss
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoProcessor
from tqdm import tqdm

from image_search.data import ImagesDataset, ConversionsDataset
from image_search.train import temporal_train_test_split
from image_search.model import QueryModel, ImageModel, LightningImageSearchSigLIP

from image_search.metrics import hit_rate, mean_average_precision_at_k

# %%
# CONSTANTS
IMAGES_FOLDER = Path("./data/unsplash-research-dataset-lite-latest/photos/")
BASE_MODEL = "google/siglip-base-patch16-224"
BATCH_SIZE = 2048
NUM_WORKERS = 4
SEED = 42

CHECKPOINT_PATHS = {
    "VANILLA": None,
    "FINE-TUNED": "./mlruns/168016125050379525/b4956e72f44f4df48b395cf338545014/checkpoints/epoch=0-step=60598.ckpt"
}

MODEL_NAME = "FINE-TUNED"
CHECKPOINT_PATH = CHECKPOINT_PATHS[MODEL_NAME]
OUTPUT_PATH = Path("./results/") / f"{MODEL_NAME}.csv"

# %%
processor = AutoProcessor.from_pretrained(BASE_MODEL)

# %%
model = AutoModel.from_pretrained(BASE_MODEL).to("cuda")

# %%
if CHECKPOINT_PATH is not None:
    # Note: Some of these keyword arguments should not be necessary after the latest changes in the code (but this would require retraining the model)
    lightning_model = LightningImageSearchSigLIP.load_from_checkpoint(CHECKPOINT_PATH, model=model, lr=1e-4)
else:
    # In case we want to load the vanilla model directly 
    lightning_model = LightningImageSearchSigLIP(model=model, lr=1e-4)

# %%
lightning_model = lightning_model.to(device="cuda", dtype=torch.bfloat16)

# %%
image_model = lightning_model.image_model
query_model = lightning_model.query_model

# %%
# Enable eval mode
image_model = image_model.eval()
query_model = query_model.eval()

# %%
torch.set_grad_enabled(False)

# %% [markdown]
# # Images

# %%
images_dataset = ImagesDataset(image_folder=IMAGES_FOLDER, processor=processor)
images_dataloader = torch.utils.data.DataLoader(images_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# %% [markdown]
# # Generate Index
# Following the [Pinecone guide](https://www.pinecone.io/learn/series/faiss/faiss-tutorial/)

# %%
embedding_dim = model.vision_model.config.hidden_size
# Note: Since this dataset contains a pretty small number of images (25k) we can just use brute-force and use a Flat Faiss index.
# This way we also avoid having to "train" the retriever.
# In a real scenario, we would try to optimize for faster retrieval with potentially millions of items
index = faiss.IndexFlatL2(embedding_dim)

# %%
for batch in tqdm(images_dataloader):
    pixel_values = batch["pixel_values"].to(device=lightning_model.device, dtype=lightning_model.dtype)
    
    image_embeddings = image_model(pixel_values=pixel_values)
    image_embeddings = image_embeddings.to(device="cpu", dtype=torch.float32).detach().numpy()

    ids = batch["id"].to("cpu").detach().numpy()
    index.add(image_embeddings)

# %%
# To avoid re-computing all indices, we will save the current index
faiss.write_index(index, f"indices/{MODEL_NAME}.index")

# %%
index = faiss.read_index(f"indices/{MODEL_NAME}.index")

# %% [markdown]
# # Dataset

# %%
# conversions = load_and_preprocess_data()
conversions = pd.read_parquet("./data/clean/conversions.parquet")
_, conversions_val = temporal_train_test_split(conversions)

# %%
# Dataset
image_dataset = ImagesDataset(image_folder=IMAGES_FOLDER, processor=processor)
dataset_val = ConversionsDataset(data=conversions_val, image_dataset=image_dataset, processor=processor)
dataloader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS)

# %%
true_ids = []
predicted_ids = []

for batch in tqdm(dataloader_val):
    input_ids = batch["input_ids"].to(lightning_model.device)
    pixel_values = batch["pixel_values"].to(device=lightning_model.device, dtype=lightning_model.dtype)

    ids = batch["ids"].unsqueeze(dim=0).numpy()

    query_embedding = query_model(input_ids)
    image_embedding = image_model(pixel_values)
    
    query_embedding = query_embedding.to(device="cpu", dtype=torch.float32).detach().numpy()
    # image_embedding = image_embedding.to("cpu").detach().numpy()

    distances, indices = index.search(query_embedding, k=25)

    true_ids.append(ids)
    predicted_ids.append(indices)

# %%
true_ids = np.concatenate(true_ids, axis=1)
predicted_ids = np.concatenate(predicted_ids, axis=0)

# %% [markdown]
# # METRICS
# - (N)DCG: https://arize.com/blog-course/ndcg/
# - (Mean) Average Precion

# %% [markdown]
# ### Hit Rate

# %%
k = [1, 5, 10, 25]

hit_rate_score = hit_rate(true_ids, predicted_ids, k=k)
map_score = mean_average_precision_at_k(true_ids=true_ids, predicted_ids=predicted_ids, k=k)

scores = pd.DataFrame({
    "hit_rate": hit_rate_score,
    "mAP": map_score,
})
scores.index.name = "k"
scores

# %%
scores.to_csv(OUTPUT_PATH)
