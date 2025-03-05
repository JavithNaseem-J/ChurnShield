import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from urllib.parse import urlparse
import numpy as np
import mlflow
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
        
        # Initialize MLflow tracking
        os.environ['MLFLOW_TRACKING_USERNAME'] = "JavithNaseem-J"
        os.environ['MLFLOW_TRACKING_PASSWORD'] = "39af2ec9240d8439ca7a568d37e4c8566f0e4507"
        
        dagshub.init(repo_owner="JavithNaseem-J", repo_name="Telecom-Customer-Churn-Prediction")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/Telecom-Customer-Churn-Prediction.mlflow")
        mlflow.set_experiment("Telecom-Customer-Churn-Prediction")

    def evaluate(self):
        try:
            # Validate file paths using Path objects for cross-platform compatibility
            if not Path(self.config.test_raw_data).exists():
                raise FileNotFoundError(f"Test data file not found at {self.config.test_raw_data}")
            if not Path(self.config.preprocessor_path).exists():
                raise FileNotFoundError(f"Preprocessor file not found at {self.config.preprocessor_path}")
            if not Path(self.config.model_path).exists():
                raise FileNotFoundError(f"Model file not found at {self.config.model_path}")

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
                logger.info(f"Loading test data from {self.config.test_raw_data}...")
                test_data = pd.read_csv(self.config.test_raw_data)
                
                # Use the original column names instead of converting to lowercase
                target_column = self.config.target_column  # Keep original case

                # Extract target column
                if target_column not in test_data.columns:
                    raise KeyError(f"Target column '{target_column}' not found in test data. Available columns: {list(test_data.columns)}")

                test_y = test_data[target_column]
                test_x = test_data.drop(columns=[target_column])

                logger.info(f"Test data shape: X={test_x.shape}, y={test_y.shape}")

                # Preprocess test features
                logger.info("Preprocessing test features...")
                test_x_transformed = preprocessor.transform(test_x)

                # Make predictions
                logger.info("Making predictions on the test data...")
                predictions = model.predict(test_x_transformed)

                # Calculate and log metrics
                logger.info("Evaluating model performance...")
                
                # Handle potential warnings for classification metrics
                try:
                    precision = precision_score(test_y, predictions, average="weighted")
                    recall = recall_score(test_y, predictions, average="weighted")
                    f1 = f1_score(test_y, predictions, average="weighted")
                except Exception as metric_error:
                    logger.warning(f"Error calculating some metrics: {str(metric_error)}. Using zero-division=0.")
                    precision = precision_score(test_y, predictions, average="weighted", zero_division=0)
                    recall = recall_score(test_y, predictions, average="weighted", zero_division=0)
                    f1 = f1_score(test_y, predictions, average="weighted", zero_division=0)
                
                metrics = {
                    "accuracy": accuracy_score(test_y, predictions),
                    "precision": precision,
                    "recall": recall,
                    "f1": f1
                }

                # Log metrics to MLflow
                mlflow.log_metrics(metrics)

                # Log model with signature
                signature = mlflow.models.infer_signature(
                    test_x_transformed, predictions
                )
                mlflow.sklearn.log_model(
                    model,
                    "TelecomCustomerChurnModel",
                    signature=signature,
                    registered_model_name="TelecomCustomerChurnModel"
                )

                logger.info(f"Model Evaluation Metrics:\naccuracy: {metrics['accuracy']}\n"
                          f"precision: {metrics['precision']}\nrecall: {metrics['recall']}\n"
                          f"f1: {metrics['f1']}")

                # Save metrics locally using Path for cross-platform compatibility
                metrics_file = Path(self.config.root_dir) / "metrics.json"
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=4)
                    
                logger.info(f"Evaluation metrics saved at {metrics_file}")

                return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise e


    
