# tests/integration_test.py

import pytest
import numpy as np
from joblib import load
from sklearn.metrics import accuracy_score
from src.models.model import create_model, compile_model
from src.models.load_parameters import load_params

# Load parameters
params = load_params()
input_folder = params["dataset_dir"] + 'processed_data/'


@pytest.fixture
def data():
    # Load data
    x_train, y_train = load(input_folder + 'ds_train.joblib')
    x_val, y_val = load(input_folder + 'ds_val.joblib')
    x_test, y_test = load(input_folder + 'ds_test.joblib')
    char_index = load(input_folder + 'char_index.joblib')

    return {
        "train_data": (x_train, y_train),
        "val_data": (x_val, y_val),
        "test_data": (x_test, y_test),
        "char_index": char_index
    }


def test_full_ml_pipeline(data):
    try:
        voc_size = len(data['char_index'].keys())
        categories = len(params['categories'])

        model = create_model(voc_size, categories)
        model = compile_model(model, params['loss_function'], params['optimizer'])

        x_train, y_train = data['train_data']
        x_val, y_val = data['val_data']

        model.fit(x_train, y_train, epochs=3, batch_size=32, validation_data=(x_val, y_val))

        x_test, y_test = data['test_data']
        y_pred = model.predict(x_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
        accuracy = accuracy_score(y_test, y_pred_labels)
        print(f"Model accuracy on test set: {accuracy}")

        assert accuracy > 0.5, "Model accuracy is below the acceptable threshold"
    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__])
