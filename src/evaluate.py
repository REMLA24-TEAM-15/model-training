import json
import os

from joblib import load
import seaborn as sns
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             accuracy_score,
                             roc_auc_score,
                             f1_score)


def evaluation():
    """
    Model evaluation
    """

    input_folder = "../../data/processed"

    predictions = load(f'{input_folder}/report.joblib')

    y_test = predictions["y_test"]
    y_pred_binary = predictions["y_pred_binary"]

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

    output_folder = "../../reports/"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    json.dump(metrics_dict, open("../../reports/metrics.json", "w"))


if __name__ == "__main__":
    evaluation()
