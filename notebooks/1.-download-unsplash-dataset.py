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

# %% [markdown]
# # IMPORANT
# This notebook is copied from [this Repo](https://github.com/haltakov/natural-language-image-search/blob/main/02-download-unsplash-dataset.ipynb).

# %%
# cd ..

# %% [markdown]
#
# # Download the Unsplash dataset
#
# This notebook can be used to download all images from the Unsplash dataset: https://github.com/unsplash/datasets. There are two versions Lite (25000 images) and Full (2M images). For the Full one you will need to apply for access (see [here](https://unsplash.com/data)). This will allow you to run CLIP on the whole dataset yourself. 
#
# Put the .TSV files in the folder `unsplash-dataset/full` or `unsplash-dataset/lite` or adjust the path in the cell below. 

# %%
from pathlib import Path

unsplash_dataset_path = Path("./data/unsplash-research-dataset-lite-latest/")

# %% [markdown]
# ## Load the dataset
#
# The `photos.tsv000` contains metadata about the photos in the dataset, but not the photos themselves. We will use the URLs of the photos to download the actual images.

# %%
import pandas as pd

# Read the photos table
photos = pd.read_csv(unsplash_dataset_path / "photos.tsv000", sep='\t', header=0)

# Extract the IDs and the URLs of the photos
photo_urls = photos[['photo_id', 'photo_image_url']].values.tolist()

# Print some statistics
print(f'Photos in the dataset: {len(photo_urls)}')

# %% [markdown]
# The file name of each photo corresponds to its unique ID from Unsplash. We will download the photos in a reduced resolution (640 pixels width), because they are downscaled by CLIP anyway.

# %%
import urllib.request

# Path where the photos will be downloaded
photos_donwload_path = unsplash_dataset_path / "photos"

# Function that downloads a single photo
def download_photo(photo):
    # Get the ID of the photo
    photo_id = photo[0]

    # Get the URL of the photo (setting the width to 640 pixels)
    photo_url = photo[1] + "?w=640"

    # Path where the photo will be stored
    photo_path = photos_donwload_path / (photo_id + ".jpg")

    # Only download a photo if it doesn't exist
    if not photo_path.exists():
        try:
            urllib.request.urlretrieve(photo_url, photo_path)
        except Exception as e:
            # Catch the exception if the download fails for some reason
            print(f"Cannot download {photo_url}.", e)
            pass


# %% [markdown]
# Now the actual download! The download can be parallelized very well, so we will use a thread pool. You may need to tune the `threads_count` parameter to achieve the optimzal performance based on your Internet connection. For me even 128 worked quite well.

# %%
# If the output folder doesn't exists, create it
photos_donwload_path.mkdir(exist_ok=True)

# %%
from multiprocessing.pool import ThreadPool

# Create the thread pool
threads_count = 16
pool = ThreadPool(threads_count)

# Start the download
pool.map(download_photo, photo_urls)

# Display some statistics
display(f'Photos downloaded: {len(photos)}')
