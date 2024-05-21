from tensorflow.keras.models import load_model
import numpy as np
from joblib import load, dump
from load_parameters import load_params


def predict():
    # Model Parameters
    params = load_params()
    data_folder = params['dataset_dir'] + "processed_data/"
    model_folder = params['dataset_dir'] + "metrics/"

    # Load data and model
    x_test, y_test = load(f'{data_folder}ds_test.joblib')
    model = load_model(model_folder + "phishing_model.h5")

    # Make predictions
    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    predictions = {"y_test": y_test, "y_pred_binary": y_pred_binary}

    dump(predictions, f'{model_folder}/predictions.joblib')


if __name__ == "__main__":
    predict()
