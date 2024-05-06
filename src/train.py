"""
train.py

Trains model based on previously loaded data.
"""
import pickle
import yaml
from dvclive import Live
from dvclive.keras import DVCLiveCallback
from model import create_model, compile_model

def main():
    """
    Trains model with parametres from params.yaml
    Model is compiled and saved.
    """

    # Model Parameters
    with open("params.yaml", encoding="utf-8") as stream:
        try:
            params = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            print("Could not load params.yaml")

    # Load Data
    data_path = params['dataset_dir']
    with open(data_path + 'small_dataset/' + 'processed_data.pkl', 'rb') as f:
        data_list = pickle.load(f)

    x_train, y_train, x_val, y_val, _, _, char_index = data_list

    # Create and Compile Model
    voc_size = len(char_index.keys())
    model = create_model(voc_size, len(params['categories']))
    model = compile_model(model, params['loss_function'], params['optimizer'])

    # Train Model
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
