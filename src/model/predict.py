import os

from keras.models import load_model
import numpy as np
from joblib import load, dump


def predict():
    """
    Model prediction
    """
    input_folder = "../../data/processed"

    # check if model and load data exists
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' is empty")

    # Load data
    x_test = load(f'{input_folder}/x_data.joblib')[2]
    y_test = load(f'{input_folder}/y_data.joblib')[2]

    model = load_model("trained_model.h5")

    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    predictions = {"y_test": y_test, "y_pred_binary": y_pred_binary}

    output_folder = "../../data/processed"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dump(predictions, f'{output_folder}/report.joblib')


if __name__ == "__main__":
    predict()
