import os
import pytest
import yaml
import numpy as np
from sklearn.metrics import classification_report, accuracy_score
from keras.api.models import load_model


@pytest.fixture
def load_test_data_and_model():
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load parameters and data
    project_directory = os.path.dirname(path) + "/model-training/"
    config_file = os.path.join(project_directory, "config.yml")

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # Check if figures directories exist:
    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/figures", exist_ok=True)

    # Load the test data
    x_test = np.load(config["processed_paths"]["x_test"])
    y_test = np.load(config["processed_paths"]["y_test"])
    y_test = y_test.reshape(-1, 1)

    # Load the trained model
    model_path = config["processed_paths"]["model_path"]
    model = load_model(model_path)

    return x_test, y_test, model


def test_model_performance(test_data_and_model):
    # Generate predictions
    x_test, y_test, model = test_data_and_model
    y_pred = model.predict(x_test, batch_size=1000)

    # Convert predicted probabilities to binary labels
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)

    report = classification_report(y_test, y_pred_binary)

    # Save accuracy to a file
    accuracy = accuracy_score(y_test, y_pred_binary)

    assert accuracy > 0.9, "Model accuracy is below 0.9"
    assert report[0]['recall'] > 0.9, "Model recall is below 0.9"
    assert report[1]['f1-score'] > 0.9, "Model f1 score is below 0.9"
