import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import numpy as np
import joblib
from mlproject.entities.config_entity import ModelEvaluationConfig
from mlproject.utils.common import save_json
from pathlib import Path


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def eval_metrics(self, actual, pred):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred, average="weighted")
        recall = recall_score(actual, pred, average="weighted")
        f1 = f1_score(actual, pred, average="weighted")
        return accuracy, precision, recall, f1

    def save_results(self):
        # Load test data and model
        test_data = pd.read_csv(self.config.test_data_path)
        model = joblib.load(self.config.model_path)

        # Split features and target
        test_x = test_data.drop(columns=[self.config.target_column])
        test_y = test_data[self.config.target_column]

        # Predict using the loaded model
        predicted_qualities = model.predict(test_x)

        # Compute metrics
        accuracy, precision, recall, f1 = self.eval_metrics(test_y, predicted_qualities)

        # Save metrics as JSON
        scores = {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        save_json(path=self.config.metric_file_path, data=scores)



    
