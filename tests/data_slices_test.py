import os
import random
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Define directories
INPUT_DIR = os.getenv("INPUT_DIR", r"datasets/raw_data/DL Dataset/")


def read_and_sample_data(file_path, sample_size=100):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        random.shuffle(lines)
        return lines[:sample_size]


@pytest.fixture
def data():
    sample_size = 100  # Set sample size for each file

    train = read_and_sample_data(INPUT_DIR + "train.txt", sample_size)
    val = read_and_sample_data(INPUT_DIR + "val.txt", sample_size)
    test = read_and_sample_data(INPUT_DIR + "test.txt", sample_size)

    # Process train data
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process validation data
    raw_x_val = [line.split("\t")[1] for line in val]
    raw_y_val = [line.split("\t")[0] for line in val]

    # Process test data
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Tokenizer setup
    tokenizer = Tokenizer(lower=True, char_level=False, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)

    # Label Encoder setup
    encoder = LabelEncoder()
    encoder.fit(raw_y_train + raw_y_val + raw_y_test)

    return {
        "train_texts": raw_x_train,
        "train_labels": raw_y_train,
        "val_texts": raw_x_val,
        "val_labels": raw_y_val,
        "test_texts": raw_x_test,
        "test_labels": raw_y_test,
        "tokenizer": tokenizer,
        "encoder": encoder,
    }


@pytest.fixture
def load_trained_model():
    model = load_model('models/phishing_model.h5')
    return model


def test_model_on_slices(data, load_trained_model):
    model = load_trained_model

    # Get the input shape expected by the model
    input_shape = model.input_shape[1:]  # Exclude batch size
    print(f"Model expected input shape: {input_shape}")

    # Determine the maximum index for the embedding layer
    max_index = model.layers[0].input_dim - 1
    print(f"Maximum index for embedding layer: {max_index}")

    # Create a representative subset (e.g., 50% of the test set)
    representative_size = int(0.5 * len(data['test_texts']))
    indices = np.random.choice(len(data['test_texts']), representative_size, replace=False)
    representative_texts = [data['test_texts'][i] for i in indices]
    representative_labels = [data['test_labels'][i] for i in indices]

    sequences = data['tokenizer'].texts_to_sequences(representative_texts)
    sequences = [[min(word_index, max_index) for word_index in sequence] for sequence in sequences]
    X_test = pad_sequences(sequences, maxlen=input_shape[0], padding='post')
    print(f"Input shape for representative slice: {X_test.shape}")
    y_true = data['encoder'].transform(representative_labels)

    y_pred = model.predict(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    accuracy = accuracy_score(y_true, y_pred_labels)
    print(f"Accuracy on representative slice: {accuracy}")

    # Check if the accuracy on the slice is within a reasonable range of the full test set accuracy
    full_sequences = data['tokenizer'].texts_to_sequences(data['test_texts'])
    full_sequences = [[min(word_index, max_index) for word_index in sequence]
                      for sequence in full_sequences]
    X_full_test = pad_sequences(full_sequences, maxlen=input_shape[0], padding='post')
    y_full_true = data['encoder'].transform(data['test_labels'])

    y_full_pred = model.predict(X_full_test)
    y_full_pred_labels = np.argmax(y_full_pred, axis=1)

    full_accuracy = accuracy_score(y_full_true, y_full_pred_labels)
    print(f"Accuracy on full test set: {full_accuracy}")

    # Allowing a slightly larger margin for statistical variations
    assert abs(
        accuracy - full_accuracy) < 0.1, "Model performance on representative slice deviates " \
                                         "significantly from full test set "


if __name__ == '__main__':
    pytest.main([__file__])
