import json
import os
from joblib import load
import seaborn as sns
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             roc_auc_score,
                             f1_score)
from .load_parameters import load_params


def evaluation(preds):
    y_test = preds["y_test"]
    y_pred_binary = preds["y_pred_binary"]

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print(f"Classification Report: {report}")

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print(f"Confusion Matrix: {confusion_mat}")
    print(f"Accuracy: {accuracy_score(y_test, y_pred_binary)}")
    sns.heatmap(confusion_mat, annot=True)

    metrics_dict = {
        "accuracy": round(accuracy_score(y_test, y_pred_binary), 5),
        "roc_auc": round(roc_auc_score(y_test, y_pred_binary), 5),
        "f1": round(f1_score(y_test, y_pred_binary), 5)
    }
    return metrics_dict


if __name__ == "__main__":
    # Model Parameters
    params = load_params()
    input_folder = params['reports_dir'] + 'metrics'

    predictions = load(f'{input_folder}/predictions.joblib')
    metric_dict = evaluation(predictions)

    output_folder = input_folder
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    with open(os.path.join(output_folder, "statistics.json"), "w") as f:
        json.dump(metric_dict, f)
