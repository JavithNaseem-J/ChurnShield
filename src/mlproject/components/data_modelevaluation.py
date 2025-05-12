import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import numpy as np
import mlflow
from mlflow.models import infer_signature
import mlflow.lightgbm
import mlflow.sklearn
import dagshub
import joblib
from src.mlproject.entities.config_entity import ModelEvaluationConfig
from src.mlproject.utils.common import save_json
from pathlib import Path
from mlproject import logger
import json



class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config


        dagshub.init(
            repo_owner="JavithNaseem-J",
            repo_name="ChurnShield"
        )
        mlflow.set_tracking_uri(
            "https://dagshub.com/JavithNaseem-J/ChurnShield.mlflow"
        )
        mlflow.set_experiment("Telecom-Customer-Churn-Prediction")

    def evaluate(self):
            mlflow.lightgbm.autolog()

            if not Path(self.config.test_raw_data).exists():
                raise FileNotFoundError(f"Test data not found at {self.config.test_raw_data}")
            if not Path(self.config.preprocessor_path).exists():
                raise FileNotFoundError(f"Preprocessor not found at {self.config.preprocessor_path}")
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Model not found at {self.config.model_path}")

            with mlflow.start_run():
                mlflow.set_tag("model_type", "CatBoostClassifier")
                mlflow.set_tag("evaluation_stage", "testing")

                logger.info("Loading preprocessor and model...")
                preprocessor = joblib.load(self.config.preprocessor_path)
                model = joblib.load(self.config.model_path)


                # Load and prepare test data
                logger.info(f"Loading test data from {self.config.test_raw_data}...")
                test_data = pd.read_csv(self.config.test_raw_data)
                target_column = self.config.target_column

                if target_column not in test_data.columns:
                    raise KeyError(f"Target column '{target_column}' not found in test data.")

                test_y = test_data[target_column]
                test_x = test_data.drop(columns=[target_column])
                logger.info(f"Test data shape: X={test_x.shape}, y={test_y.shape}")

                logger.info("Preprocessing test features...")
                test_x_transformed = preprocessor.transform(test_x)

                logger.info("Making predictions...")
                predictions = model.predict(test_x_transformed)

                logger.info("Evaluating model performance...")

                precision = precision_score(test_y, predictions, average="weighted")

                recall = recall_score(test_y, predictions, average="weighted")
                
                f1 = f1_score(test_y, predictions, average="weighted")

                metrics = {
                    "accuracy": accuracy_score(test_y, predictions),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

                mlflow.log_metrics(metrics)

                signature = infer_signature(test_x_transformed, predictions)
                mlflow.sklearn.log_model(
                    model,
                    artifact_path="TelecomCustomerChurnModel",
                    signature=signature,
                    registered_model_name="TelecomCustomerChurnModel"
                )

                logger.info(f"Evaluation Metrics:\n{json.dumps(metrics, indent=2)}")
                metrics_file = Path(self.config.root_dir) / "metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Metrics saved to {metrics_file}")

                return metrics


    
