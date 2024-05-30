from joblib import load

from dvclive import Live
from dvclive.keras import DVCLiveCallback
from .model import create_model, compile_model
from .load_parameters import load_params


def main():
    # Model Parameters
    params = load_params()

    # Load data
    input_folder = params["dataset_dir"] + 'processed_data/'
    x_train, y_train = load(input_folder + 'ds_train.joblib')
    x_val, y_val = load(input_folder + 'ds_val.joblib')
    char_index = load(input_folder + 'char_index.joblib')

    # Create and Compile Model
    voc_size = len(char_index.keys())
    model = create_model(voc_size, len(params['categories']))
    model = compile_model(model, params['loss_function'], params['optimizer'])

    # Train Model
    metrics_path = params['dataset_dir'] + 'metrics/'
    with Live(metrics_path, dvcyaml="dvc.yaml") as live:
        model.fit(x_train, y_train,
                  batch_size=params['batch_train'],
                  epochs=params['epoch'],
                  shuffle=True,
                  validation_data=(x_val, y_val),
                  callbacks=[DVCLiveCallback(live=live)]
                  )
        # Save Model
        model.save(metrics_path + 'phishing_model.h5')
        live.log_artifact(metrics_path + "phishing_model.h5", type='model')
    print("Trained model saved")


if __name__ == "__main__":
    main()
