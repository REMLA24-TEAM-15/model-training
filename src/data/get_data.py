import sys
sys.path.insert(0, '.')

"""
This module downloads the dataset from kaggle and stores it in the data folder.
"""
import os
import kaggle
from src.models.load_parameters import load_params


def get_data():
    # Model Parameters
    params = load_params()

    # Set the folder path
    folder_path = params['dataset_dir'] + "raw_data/"

    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.chmod("../..", 0o777)
        os.makedirs(folder_path)

    # Download the dataset using the Kaggle API
    kaggle.api.dataset_download_files('aravindhannamalai/dl-dataset', path=folder_path, unzip=True)


if __name__ == '__main__':
    get_data()
