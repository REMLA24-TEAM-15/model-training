# evaluate.py
from model import create_model
from data_prep import preprocess_data, load_data
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import json

def main():
    # Load Data
    test_file = "data/raw/DL Dataset/test.txt"
    raw_x_test, raw_y_test = load_data(test_file)

    # Preprocess Data
    x_test, y_test, _ = preprocess_data([], [], [], [], raw_x_test, raw_y_test, char_index)

    # Load Model
    model = create_model(voc_size=len(char_index.keys()))
    model.load_weights('phishing_model.h5')

    # Evaluate Model
    y_pred = model.predict(x_test, batch_size=1000)
    y_pred_binary = (np.array(y_pred) > 0.5).astype(int)
    y_test = y_test.reshape(-1, 1)

    # Calculate classification report
    report = classification_report(y_test, y_pred_binary)
    print('Classification Report:')
    print(report)

    # Calculate confusion matrix
    confusion_mat = confusion_matrix(y_test, y_pred_binary)
    print('Confusion Matrix:', confusion_mat)
    print('Accuracy:', accuracy_score(y_test, y_pred_binary))

    # Plot Confusion Matrix
    sns.heatmap(confusion_mat, annot=True)
    plt.show()

    metrics_dict = {
        "accuracy": round(accuracy_score(y_test, y_pred_binary), 5)
    }
    with open("../../reports/metrics.json", "w", encoding="utf-8") as json_file:
        json.dump(metrics_dict, json_file, indent=4)

if __name__ == "__main__":
    main()
