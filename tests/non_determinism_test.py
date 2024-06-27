# tests/test_non_determinism.py

import pytest
import numpy as np
import random
import tensorflow as tf
from joblib import load
from src.models.model import create_model, compile_model
from src.models.load_parameters import load_params

# tests/test_non_determinism.py

import pytest
import numpy as np
import random
import tensorflow as tf
from joblib import load
from src.models.model import create_model, compile_model
from src.models.load_parameters import load_params


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


def evaluate_score(model, x_val, y_val):
    # Evaluate the model and return a performance score
    loss, accuracy = model.evaluate(x_val, y_val, verbose=0)
    return accuracy  # or any other performance metric


def test_non_determinism(model_params, datasets):
    (x_train, y_train), (x_val, y_val) = datasets

    # Set random seeds for reproducibility
    set_seeds(42)

    # Create and train the model twice with the same random seed
    voc_size = len(load(model_params["dataset_dir"] + 'processed_data/char_index.joblib').keys())

    set_seeds(42)
    model_1 = create_model(voc_size, len(model_params['categories']))
    model_1 = compile_model(model_1, model_params['loss_function'], model_params['optimizer'])

    set_seeds(42)
    model_2 = create_model(voc_size, len(model_params['categories']))
    model_2 = compile_model(model_2, model_params['loss_function'], model_params['optimizer'])

    # Use a smaller batch size for testing
    batch_size = 32  # Adjust batch size as necessary

    # Train both models
    model_1.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)
    model_2.fit(x_train, y_train, epochs=1, batch_size=batch_size, verbose=0)

    # Evaluate both models
    score_1 = evaluate_score(model_1, x_val, y_val)
    score_2 = evaluate_score(model_2, x_val, y_val)

    # Compare performance
    assert abs(score_1 - score_2) <= 0.03, f"Model performance is not consistent: {score_1} vs {score_2}"


if __name__ == "__main__":
    pytest.main()
