
from keras.models import load_model
import yaml
from joblib import load

def train_model():
    """
    Train model
    """
    input_folder = "../../data/processed"
    model = load_model("model.h5")
    with open("../../configs/params.yaml", "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    x = load(f'{input_folder}/x_data.joblib')
    y = load(f'{input_folder}/y_data.joblib')

    x_train = x[0]
    y_train = y[0]
    x_val = x[1]
    y_val = y[1]

    model.compile(
        loss=params["loss_function"],
        optimizer=params["optimizer"],
        metrics=["accuracy"],
    )

    model.fit(
        x_train,
        y_train,
        batch_size=params["batch_train"],
        epochs=params["epoch"],
        shuffle=True,
        validation_data=(x_val, y_val),
    )

    model.save("trained_model.h5")


if __name__ == "__main__":
    train_model()
