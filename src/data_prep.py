#data preprocessing

import os
from joblib import dump

# tokenizing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder


def split_dataset(dataset_folder="../../data/raw/DL Dataset/", train_file="train.txt",
                  test_file="test.txt", val_file="val.txt"):

    with open(dataset_folder + train_file, "r", encoding="utf-8") as train_lines:
        train = [line.strip() for line in train_lines.readlines()]
        raw_x_train = [line.split("\t")[1] for line in train]
        raw_y_train = [line.split("\t")[0] for line in train]

    with open(dataset_folder + test_file, "r", encoding="utf-8") as test_lines:
        test = [line.strip() for line in test_lines.readlines()]
        raw_x_test = [line.split("\t")[1] for line in test]
        raw_y_test = [line.split("\t")[0] for line in test]

    with open(dataset_folder + val_file, "r", encoding="utf-8") as val_lines:
        val = [line.strip() for line in val_lines.readlines()]
        raw_x_val = [line.split("\t")[1] for line in val]
        raw_y_val = [line.split("\t")[0] for line in val]

    print("Train size: ", len(raw_x_train))
    print("Test size: ", len(raw_x_test))
    print("Val size: ", len(raw_x_val))

    return [raw_x_train, raw_x_test, raw_x_val], [raw_y_train, raw_y_test, raw_y_val]


def tokenizing(raw_x, raw_y):
    """Tokenizes the raw data."""
    raw_x_train, raw_x_test, raw_x_val = raw_x
    raw_y_train, raw_y_test, raw_y_val = raw_y

    tokenizer = Tokenizer(lower=True, char_level=True, oov_token='-n-')
    tokenizer.fit_on_texts(raw_x_train + raw_x_val + raw_x_test)
    char_index = tokenizer.word_index
    sequence_length = 200
    x_train = pad_sequences(tokenizer.texts_to_sequences(raw_x_train), maxlen=sequence_length)
    x_val = pad_sequences(tokenizer.texts_to_sequences(raw_x_val), maxlen=sequence_length)
    x_test = pad_sequences(tokenizer.texts_to_sequences(raw_x_test), maxlen=sequence_length)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(raw_y_train)
    y_val = encoder.transform(raw_y_val)
    y_test = encoder.transform(raw_y_test)

    return char_index, [x_train, x_val, x_test], [y_train, y_val, y_test]


# main function
def main():

    dataset_folder = "../../data/raw/DL Dataset/"

    # check if there is data inside the folder
    if not os.path.exists(dataset_folder):
        raise FileNotFoundError(f"Dataset folder '{dataset_folder}' is empty")

    raw_x, raw_y = split_dataset()
    char_index, tokenized_x, tokenized_y = tokenizing(raw_x, raw_y)

    output_folder = "../../data/processed"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    dump(char_index, f'{output_folder}/char_index.joblib')
    dump(tokenized_x, f'{output_folder}/x_data.joblib')
    dump(tokenized_y, f'{output_folder}/y_data.joblib')

if __name__ == "__main__":
    main()