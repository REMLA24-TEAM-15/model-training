import numpy as np
import pytest
import random
import tensorflow as tf
from joblib import load
from src.models.load_parameters import load_params
from src.models.model import *


# Function to set seeds for reproducibility
def set_seeds(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


@pytest.fixture
def model_params():
    return load_params()


@pytest.fixture
def datasets(model_params):
    input_folder = model_params["dataset_dir"] + 'processed_data/'
    x_train, y_train = load(input_folder + 'ds_train.joblib')
    x_val, y_val = load(input_folder + 'ds_val.joblib')
    return (x_train, y_train), (x_val, y_val)


@pytest.fixture
def model(model_params):
    voc_size = len(load(model_params["dataset_dir"] + 'processed_data/char_index.joblib').keys())
    model = create_model(voc_size, len(model_params['categories']))
    return compile_model(model, model_params['loss_function'], model_params['optimizer'])


def test_non_determinism(model_params, datasets):
    (x_train, y_train), _ = datasets

    # Set random seeds for reproducibility
    set_seeds(42)

    # Create and train the model twice
    voc_size = len(load(model_params["dataset_dir"] + 'processed_data/char_index.joblib').keys())

    set_seeds(42)
    model_1 = create_model(voc_size, len(model_params['categories']))
    model_1 = compile_model(model_1, model_params['loss_function'], model_params['optimizer'])

    set_seeds(42)
    model_2 = create_model(voc_size, len(model_params['categories']))
    model_2 = compile_model(model_2, model_params['loss_function'], model_params['optimizer'])

    # Train both models
    model_1.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
    model_2.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)

    # Compare weights
    weights_1 = model_1.get_weights()
    weights_2 = model_2.get_weights()

    for w1, w2 in zip(weights_1, weights_2):
        print(w1, w2)

    for w1, w2 in zip(weights_1, weights_2):
        assert np.allclose(w1, w2), "Model weights are not consistent between training runs"


if __name__ == "__main__":
    pytest.main()
