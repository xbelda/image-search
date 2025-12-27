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
# cd ..

# %%
from pathlib import Path

import polars as pl

from image_search.file_cleaning import transform_collections, transform_colors, transform_conversions, transform_keywords, transform_photos

# %%
transforms = {
    "collections": transform_collections,
    "conversions": transform_conversions,
    "keywords": transform_keywords,
    "photos": transform_photos,
    "colors": transform_colors
}

# %%
original_folder = Path("./data/unsplash-research-dataset-lite-latest/")
dest_folder = Path("./data/clean/")

dest_folder.mkdir(exist_ok=True)

# %%
for p in original_folder.glob("*.csv000"):
    filename = p.stem
    print(filename)

    df = pl.read_csv(p, separator="\t", infer_schema_length=1000)
    
    transform = transforms[filename]

    df = transform(df)

    output_path = dest_folder / f"{filename}.parquet"
    df.write_parquet(output_path)
