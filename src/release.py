from joblib import load, dump
from tensorflow.keras.models import load_model
from src.models.load_parameters import load_params

if __name__ == '__main__':
    # Model Parameters
    params = load_params()

    input_folder = params["dataset_dir"] + 'processed_data/'
    char_index = load(input_folder + 'char_index.joblib')

    metrics_path = params['dataset_dir'] + 'metrics/'
    model = load_model(metrics_path + 'phishing_model.h5')
    version = params['version']

    out_dict = {
        'version': version,
        'char_index': char_index,
        'model': model
    }
    dump(out_dict, 'release.joblib')
    print("Exit")
