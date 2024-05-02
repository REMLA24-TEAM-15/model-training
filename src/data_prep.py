# data_preprocessing.py
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def load_data(file_path):
    data = [line.strip() for line in open(file_path, "r").readlines()]
    x_data = [line.split("\t")[1] for line in data]
    y_data = [line.split("\t")[0] for line in data]
    return x_data, y_data

def preprocess_data(raw_x_train, raw_y_train, raw_x_val, raw_y_val, raw_x_test, raw_y_test):
    # Tokenization and Padding
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    # Encoding Labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return x_train, y_train, x_val, y_val, x_test, y_test, char_index
