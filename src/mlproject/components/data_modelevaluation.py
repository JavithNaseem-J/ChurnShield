import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
import joblib
from mlproject.entities.config_entity import ModelEvaluationConfig
from mlproject.utils.common import save_json
from pathlib import Path
from mlproject import logger
import json


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        # Validate file paths
        if not os.path.exists(self.config.test_raw_data):
            raise FileNotFoundError(f"Test data file not found at {self.config.test_raw_data}")
        if not os.path.exists(self.config.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {self.config.preprocessor_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model file not found at {self.config.model_path}")

        # Load preprocessor and model
        logger.info("Loading preprocessor and model...")
        preprocessor = joblib.load(self.config.preprocessor_path)
        model = joblib.load(self.config.model_path)

        # Load test data
        logger.info(f"Loading test data from {self.config.test_raw_data}...")
        test_data = pd.read_csv(self.config.test_raw_data)

        # Extract target column
        if self.config.target_column not in test_data.columns:
            raise KeyError(f"Target column '{self.config.target_column}' not found in test data")

        test_y = test_data[self.config.target_column]
        test_x = test_data.drop(columns=[self.config.target_column])

        logger.info(f"Test data shape: X={test_x.shape}, y={test_y.shape}")

        # Preprocess test features
        logger.info("Preprocessing test features...")
        test_x_transformed = preprocessor.transform(test_x)

        # Make predictions
        logger.info("Making predictions on the test data...")
        predictions = model.predict(test_x_transformed)

        # Evaluate the model
        logger.info("Evaluating model performance...")
        accuracy = accuracy_score(test_y, predictions)
        precision = precision_score(test_y, predictions)
        recall = recall_score(test_y, predictions)
        f1 = f1_score(test_y, predictions)

        logger.info(f"Model Evaluation Metrics: \n accuracy_score: {accuracy} \n precision_score: {precision} \n recall_score: {recall} \n f1_score: {f1}")
  
        # Save the evaluation metrics
        metrics_path = os.path.join(self.config.root_dir, "metrics.json")
        metrics = {
            "accuracy_score": accuracy,
            "precision_score": precision,
            "recall_score": recall,
            "f1_score": f1
        }
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        logger.info(f"Evaluation metrics saved at {metrics_path}")

        return metrics
    

'''class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        
        # Initialize MLflow tracking
        os.environ['MLFLOW_TRACKING_USERNAME'] = "JavithNaseem-J"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "fc8980ed3562ad9f63b1351811c99954c11dd06b"
        
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="Wine-Quality-Prediction")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/Wine-Quality-Prediction.mlflow")
        mlflow.set_experiment("wine_quality_prediction")

    def evaluate(self):
        try:

            with mlflow.start_run():
                # Set tags for the run
                mlflow.set_tag("model_type", "CatBoostClassifier")
                mlflow.set_tag("evaluation_stage", "testing")

                # Load preprocessor and model
                logger.info("Loading preprocessor and model...")
                preprocessor = joblib.load(self.config.preprocessor_path)
                model = joblib.load(self.config.model_path)

                # Log model parameters
                mlflow.log_params(self.config.all_params)

                # Load test data
                logger.info(f"Loading test data from {self.config.test_data_path}...")
                test_data = pd.read_csv(self.config.test_data_path)

                # Extract target column
                if self.config.target_column not in test_data.columns:
                    raise KeyError(f"Target column '{self.config.target_column}' not found in test data")

                test_y = test_data[self.config.target_column]
                test_x = test_data.drop(columns=[self.config.target_column])

                logger.info(f"Test data shape: X={test_x.shape}, y={test_y.shape}")

                # Preprocess test features
                logger.info("Preprocessing test features...")
                test_x_transformed = preprocessor.transform(test_x)

                # Make predictions
                logger.info("Making predictions on the test data...")
                predictions = model.predict(test_x_transformed)

                # Calculate and log metrics
                logger.info("Evaluating model performance...")
                metrics = {
                    "accuracy": accuracy_score(test_y, predictions),
                    "precision": precision_score(test_y, predictions, average="weighted"),
                    "recall": recall_score(test_y, predictions, average="weighted"),
                    "f1": f1_score(test_y, predictions, average="weighted")
                }

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Log model with signature
                signature = mlflow.models.infer_signature(
                    test_x_transformed, predictions
                )
                mlflow.sklearn.log_model(
                    model,
                    "wine_quality_model",
                    signature=signature,
                    registered_model_name="WineQualityClassifier"
                )

                logger.info(f"Model Evaluation Metrics:\naccuracy: {metrics['accuracy']}\n"
                          f"precision: {metrics['precision']}\nrecall: {metrics['recall']}\n"
                          f"f1: {metrics['f1']}")

                # Save metrics locally
                metrics_path = os.path.join(self.config.root_dir, "metrics.json")
                with open(metrics_path, "w") as f:
                    json.dump(metrics, f, indent=4)
                logger.info(f"Evaluation metrics saved at {metrics_path}")

                return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e'''



    
