"""
train.py

Trains model based on previously loaded data.
"""

# train.py
from data_preprocessing import preprocess_data, load_data
from model import create_model, compile_model

# Load Data
TRAIN_FILE = "/kaggle/input/dl-dataset/DL Dataset/train.txt"
VAL_FILE = "/kaggle/input/dl-dataset/DL Dataset/val.txt"
TEST_FILE = "/kaggle/input/dl-dataset/DL Dataset/test.txt"

raw_x_train, raw_y_train = load_data(TRAIN_FILE)
raw_x_val, raw_y_val = load_data(VAL_FILE)
raw_x_test, raw_y_test = load_data(TEST_FILE)

# Preprocess Data
x_train, y_train, x_val, y_val, x_test, y_test, char_index = preprocess_data(raw_x_train, raw_y_train, raw_x_val, raw_y_val, raw_x_test, raw_y_test)

# Model Parameters
params = {'loss_function': 'binary_crossentropy',
          'optimizer': 'adam',
          'sequence_length': 200,
          'batch_train': 5000,
          'batch_test': 5000,
          'categories': ['phishing', 'legitimate'],
          'char_index': None,
          'epoch': 30,
          'embedding_dimension': 50,
          'dataset_dir': "../dataset/small_dataset/"}

# Create and Compile Model
voc_size = len(char_index.keys())
model = create_model(voc_size)
model = compile_model(model, params['loss_function'], params['optimizer'])

# Train Model
hist = model.fit(x_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(x_val, y_val)
                )

# Save Model
model.save('phishing_model.h5')
