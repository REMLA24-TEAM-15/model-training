"""
Prepares data for further processing.
"""

# data_preprocessing.py
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(file_path):
    """
    Loads data from a file.
    Returns list of labels and list of datapoints.
    """ 
    with open(file_path, "r", encoding="utf-8") as file:
        data = [line.strip() for line in file.readlines()]
    x_data = [line.split("\t")[1] for line in data]
    y_data = [line.split("\t")[0] for line in data]
    return x_data, y_data

def preprocess_data(raw_train, raw_val, raw_test):
    """
    Tokenizes and pads data. Encodes labels.
    """
    # Tokenization and Padding
    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_train[0] + raw_val[0] + raw_test[0])
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_train[0]), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_val[0]), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_test[0]), maxlen=sequence_length)

    # Encoding Labels
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(raw_train[1])
    y_val = encoder.transform(raw_val[1])
    y_test = encoder.transform(raw_test[1])

    return x_train, y_train, x_val, y_val, x_test, y_test, char_index
