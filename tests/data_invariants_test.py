import os
import random
import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from scipy.stats import ks_2samp

# Define directories
INPUT_DIR = os.getenv("INPUT_DIR", r"datasets/raw_data/DL Dataset/")

def read_data(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        lines = [line.strip() for line in file.readlines()]
        return lines

@pytest.fixture
def data():
    # Read all data
    train = read_data(INPUT_DIR + "train.txt")
    test = read_data(INPUT_DIR + "test.txt")

    # Process train data
    raw_x_train = [line.split("\t")[1] for line in train]
    raw_y_train = [line.split("\t")[0] for line in train]

    # Process test data
    raw_x_test = [line.split("\t")[1] for line in test]
    raw_y_test = [line.split("\t")[0] for line in test]

    # Tokenizer setup
    tokenizer = Tokenizer(lower=True, char_level=False, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_test)

    return {
        "train_texts": raw_x_train,
        "train_labels": raw_y_train,
        "test_texts": raw_x_test,
        "test_labels": raw_y_test,
        "tokenizer": tokenizer,
    }

def test_data_invariants(data):
    # Tokenize the texts
    train_sequences = data['tokenizer'].texts_to_sequences(data['train_texts'])
    test_sequences = data['tokenizer'].texts_to_sequences(data['test_texts'])

    # Flatten the sequences to get a distribution of tokens
    train_tokens = [token for sequence in train_sequences for token in sequence]
    test_tokens = [token for sequence in test_sequences for token in sequence]

    # Compare distributions using the Kolmogorov-Smirnov test
    ks_stat, p_value = ks_2samp(train_tokens, test_tokens)

    print(f"KS Statistic: {ks_stat}, P-Value: {p_value}")

    # Assert that distributions are not significantly different
    assert p_value > 0.05, "Distributions of tokens in training and serving data are significantly different."

if __name__ == '__main__':
    pytest.main([__file__])
