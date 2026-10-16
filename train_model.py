"""
train_model.py - trains the Isolation Forest and saves it to models/anomaly_model.pkl

Run:  python train_model.py

Steps: load data -> clean -> derive features -> split train/test ->
       train on the training part only -> evaluate on the test part -> save.
"""

from src.anomaly_detection import train_and_save
from src.config import DATA_PATH, MODEL_PATH
from src.data_processing import DataValidationError


def main():
    try:
        bundle, metrics = train_and_save()
    except DataValidationError as error:
        print(f"Training failed: {error}")
        return

    print(f"Dataset used : {DATA_PATH}")
    print(f"Rows trained : {bundle['trained_rows']}")
    print(f"Model saved  : {MODEL_PATH}")
    if metrics:
        print(f"Test rows    : {metrics['test_rows']}")
        print(f"Precision    : {metrics['precision']:.2f}  (flagged rows that were truly anomalous)")
        print(f"Recall       : {metrics['recall']:.2f}  (true anomalies that were found)")


if __name__ == "__main__":
    main()
