import os
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import yaml
from joblib import load


def model():
    input_folder = "../../data/processed"

    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Input folder '{input_folder}' is empty")

    char_index = load(f'{input_folder}/char_index.joblib')

    with open("../../configs/params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    model = Sequential()
    voc_size = len(char_index.keys())
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation="tanh"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 7, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 5, activation="tanh", padding="same"))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Conv1D(128, 3, activation="tanh", padding="same"))
    model.add(MaxPooling1D(3))
    model.add(Dropout(0.2))

    model.add(Flatten())

    model.add(Dense(len(params["categories"]) - 1, activation="sigmoid"))

    model.save("model.h5")


if __name__ == "__main__":
    model()
