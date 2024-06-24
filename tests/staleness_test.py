import pytest
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import accuracy_score
import os
import random
from src.models.model import create_model, compile_model

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
def load_production_model():
    # Load the current production model
    model = load_model('models/phishing_model.h5')
    return model

def test_model_staleness(data, load_production_model):
    production_model = load_production_model

    # Compile the production model
    production_model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    # Train a new model with the latest data
    max_sequence_length = 100
    vocab_size = len(data['tokenizer'].word_index) + 1  # +1 for the OOV token
    categories = len(data['encoder'].classes_)

    new_model = create_model(vocab_size, categories)
    new_model = compile_model(new_model, loss_function='binary_crossentropy', optimizer=Adam(learning_rate=0.001))

    X_train = pad_sequences(data['tokenizer'].texts_to_sequences(data['train_texts']), maxlen=max_sequence_length)
    y_train = data['encoder'].transform(data['train_labels'])

    print(f"X_train shape: {X_train.shape}")
    print(f"Sample X_train sequences: {X_train[:5]}")
    print(f"y_train shape: {y_train.shape}")
    print(f"Sample y_train: {y_train[:5]}")

    new_model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.1)

    # Evaluate both models on the validation dataset
    X_val = pad_sequences(data['tokenizer'].texts_to_sequences(data['val_texts']), maxlen=max_sequence_length)
    y_val = data['encoder'].transform(data['val_labels'])

    print(f"X_val shape: {X_val.shape}")
    print(f"Sample X_val sequences: {X_val[:5]}")
    print(f"y_val shape: {y_val.shape}")
    print(f"Sample y_val: {y_val[:5]}")

    try:
        production_y_pred = production_model.predict(X_val)
        production_y_pred_labels = (production_y_pred > 0.5).astype(int)
        production_accuracy = accuracy_score(y_val, production_y_pred_labels)

        new_y_pred = new_model.predict(X_val)
        new_y_pred_labels = (new_y_pred > 0.5).astype(int)
        new_accuracy = accuracy_score(y_val, new_y_pred_labels)

        print(f"Production model accuracy: {production_accuracy}")
        print(f"New model accuracy: {new_accuracy}")

        assert production_accuracy >= new_accuracy, "The production model is stale. The new model outperforms it significantly."
    except Exception as e:
        print(f"Error during model prediction: {e}")
        raise e

if __name__ == '__main__':
    pytest.main([__file__])
