import sys
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from joblib import dump
from src.models.load_parameters import load_params

sys.path.insert(0, '.')


def load_data(file_path):
    with open(file_path, "r") as file:
        data = [line.strip() for line in file.readlines()]
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

    return [x_train, y_train], [x_val, y_val], [x_test, y_test], char_index


def main():
    # Model Parameters
    params = load_params()

    data_path = params['dataset_dir'] + 'raw_data/DL Dataset/'

    train_file = data_path + "train.txt"
    val_file = data_path + "val.txt"
    test_file = data_path + "test.txt"

    raw_x_train, raw_y_train = load_data(train_file)
    raw_x_val, raw_y_val = load_data(val_file)
    raw_x_test, raw_y_test = load_data(test_file)

    ds_train, ds_val, ds_test, char_index = preprocess_data(raw_x_train, raw_y_train,
                                                            raw_x_val, raw_y_val,
                                                            raw_x_test, raw_y_test)

    out_path = params['dataset_dir'] + 'processed_data/'
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    dump(ds_train, f'{out_path}ds_train.joblib')
    dump(ds_val, f'{out_path}ds_val.joblib')
    dump(ds_test, f'{out_path}ds_test.joblib')
    dump(char_index, f'{out_path}char_index.joblib')

    print("Done preprocessing data. Exiting data_prep.py")


if __name__ == "__main__":
    main()
