# tests/features_cost_test.py

import pytest
from joblib import load
from src.models.model import create_model, compile_model
from src.models.load_parameters import load_params

MAX_PARAMETERS = 1_000_000  # Example threshold for max parameters

@pytest.fixture
def data():
    # Load parameters
    params = load_params()
    input_folder = params["dataset_dir"] + 'processed_data/'

    # Load data
    x_train, y_train = load(input_folder + 'ds_train.joblib')
    char_index = load(input_folder + 'char_index.joblib')

    return {
        "x_train": x_train,
        "y_train": y_train,
        "char_index": char_index,
        "params": params
    }

def test_cost_of_features(data):
    x_train, y_train, char_index, params = data.values()

    # Create and compile model
    voc_size = len(char_index.keys())
    model = create_model(voc_size, len(params['categories']))
    model = compile_model(model, params['loss_function'], params['optimizer'])

    # Build the model with input shape to count parameters
    model.build(input_shape=(None, params['sequence_length']))

    # Check the number of parameters
    num_params = model.count_params()

    print(f"Number of parameters in the model: {num_params}")
    assert num_params < MAX_PARAMETERS, f"Model has too many parameters: {num_params} (Limit: {MAX_PARAMETERS})"

if __name__ == "__main__":
    pytest.main([__file__])
