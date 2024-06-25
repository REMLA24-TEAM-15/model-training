# tests/memory_usage_test.py

import os
import time
import psutil
import pytest
from joblib import load
from multiprocessing import Process, Queue
from src.models.model import create_model, compile_model
from src.models.load_parameters import load_params

MAX_MEMORY_USAGE_MB = 250


def memory_monitor(queue):
    process = psutil.Process(os.getpid())
    max_memory = 0
    while True:
        memory = process.memory_info().rss
        if memory > max_memory:
            max_memory = memory
        if not queue.empty() and queue.get() == 'STOP':
            break
        time.sleep(0.1)
    queue.put(max_memory)


def measure_memory_usage(func, *args, **kwargs):
    queue = Queue()
    p = Process(target=memory_monitor, args=(queue,))
    p.start()

    result = func(*args, **kwargs)

    queue.put('STOP')
    max_memory = queue.get()
    p.join()

    return max_memory, result


@pytest.fixture
def data():
    # Load parameters
    params = load_params()
    input_folder = params["dataset_dir"] + 'processed_data/'

    # Load data
    x_train, y_train = load(input_folder + 'ds_train.joblib')
    x_val, y_val = load(input_folder + 'ds_val.joblib')
    char_index = load(input_folder + 'char_index.joblib')

    # Reduce the size of the dataset for faster testing
    x_train = x_train[:100]
    y_train = y_train[:100]
    x_val = x_val[:20]
    y_val = y_val[:20]

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_val": x_val,
        "y_val": y_val,
        "char_index": char_index,
        "params": params
    }


def train_model(data):
    x_train, y_train, x_val, y_val, char_index, params = data.values()

    # Create and compile model
    voc_size = len(char_index.keys())
    model = create_model(voc_size, len(params['categories']))
    model = compile_model(model, params['loss_function'], params['optimizer'])

    # Train model
    model.fit(x_train, y_train, epochs=params['epoch'], batch_size=params['batch_train'],
              validation_data=(x_val, y_val))
    return model


def test_memory_usage(data):
    max_memory_usage, _ = measure_memory_usage(train_model, data)
    max_memory_usage_mb = max_memory_usage / 1024 / 1024  # Convert to MB

    assert max_memory_usage_mb < MAX_MEMORY_USAGE_MB, f"Maximum memory usage exceeded: {max_memory_usage_mb:.2f} MB (Limit: {MAX_MEMORY_USAGE_MB} MB)"


if __name__ == "__main__":
    pytest.main([__file__])
