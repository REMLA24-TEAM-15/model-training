"""
This module downloads the dataset from kaggle and stores it in the data folder.
"""
import os
import kaggle

# Set the folder path
FOLDER_PATH = "../data/raw"

# Create the folder if it doesn't exist
if not os.path.exists(FOLDER_PATH):
    os.chmod(".", 0o777)
    os.makedirs(FOLDER_PATH)

# Download the dataset using the Kaggle API
kaggle.api.dataset_download_files('aravindhannamalai/dl-dataset', path=FOLDER_PATH, unzip=True)
