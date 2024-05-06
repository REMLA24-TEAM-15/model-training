"""
train.py

Trains model based on previously loaded data.
"""
from model import create_model, compile_model
import yaml
import pickle
from dvclive import Live
from dvclive.keras import DVCLiveCallback


def main():
    # Model Parameters
    with open("params.yaml") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            raise "Could not load params.yaml"

    # Load Data
    data_path = params['dataset_dir']
    with open(data_path + 'small_dataset/' + 'processed_data.pkl', 'rb') as f:
        data_list = pickle.load(f)

    x_train, y_train, x_val, y_val, x_test, y_test, char_index = data_list

    # Create and Compile Model
    voc_size = len(char_index.keys())
    model = create_model(voc_size, len(params['categories']))
    model = compile_model(model, params['loss_function'], params['optimizer'])

    # Train Model
    #todo: Remove 10000
    with Live(data_path + 'metrics/', dvcyaml="dvc.yaml") as live:
        model.fit(x_train[:10000], y_train[:10000],
                  batch_size=params['batch_train'],
                  epochs=params['epoch'],
                  shuffle=True,
                  validation_data=(x_val[:10000], y_val[:10000]),
                  callbacks=[DVCLiveCallback(live=live)]
                  )

    # Save Model
    model.save(data_path + 'phishing_model.h5')


if __name__ == "__main__":
    main()
