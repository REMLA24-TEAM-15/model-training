import yaml
from joblib import load, dump
from tensorflow.keras.models import load_model

if __name__ == '__main__':
    # Model Parameters
    with open("params.yaml") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise "Could not load params.yaml"

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
