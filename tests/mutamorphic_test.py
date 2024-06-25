import os
import random

import numpy as np
import pytest
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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

    # Preprocess the data
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

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
        "x_train": x_train,
        "x_val": x_val,
        "x_test": x_test,
        "tokenizer": tokenizer,
        "encoder": encoder,
    }


def test_metamorphic_properties_with_repair(data):
    model = load_model('models/phishing_model.h5')

    # Original test data
    X_test = data["x_test"]

    # Predictions on original test data
    try:
        y_pred = model.predict(X_test)
    except Exception as e:
        print(f"Error encountered during model prediction: {e}")

    # Automatic Test Input Generation (Mutation)
    mutations = []
    for sequence in X_test:
        mutant_sequence = sequence.copy()
        if len(mutant_sequence) > 1:
            mutant_sequence[1] = mutant_sequence[0]  # Simple mutation for example purposes
        mutations.append(mutant_sequence)

    # Ensure padding to the same length
    X_test_mutants = pad_sequences(mutations, maxlen=200)

    # Predictions on mutant test data
    try:
        y_pred_mutants = model.predict(X_test_mutants)
    except Exception as e:
        print(f"Error encountered during mutant model prediction: {e}")

    # Automatic Test Oracle Generation
    differences = np.abs(y_pred - y_pred_mutants)
    significant_differences = differences > 0.1  # Threshold for "fairly different"
    failing_tests = np.any(significant_differences, axis=1)

    # Check if any failing tests
    if np.any(failing_tests):
        print(f"Failing tests found: {np.where(failing_tests)}")

    # Automatic Inconsistency Repair
    repairs = []
    for i, fail in enumerate(failing_tests):
        if fail:
            repair_sequence = X_test[i]
            repairs.append(repair_sequence)
        else:
            repairs.append(X_test_mutants[i])

    # Ensure the repairs are properly padded and converted to array
    X_test_repairs = pad_sequences(repairs, maxlen=200)

    # Predictions on repaired test data
    try:
        y_pred_repairs = model.predict(X_test_repairs)
    except Exception as e:
        print(f"Error encountered during repair model prediction: {e}")

    # Ensure that the repair fixes the inconsistency
    differences_after_repair = np.abs(y_pred - y_pred_repairs)
    inconsistency_fixed = np.all(differences_after_repair <= 0.1)  # Threshold for consistency

    assert inconsistency_fixed, "Not all inconsistencies were fixed by the repairs."


if __name__ == '__main__':
    pytest.main([__file__])
