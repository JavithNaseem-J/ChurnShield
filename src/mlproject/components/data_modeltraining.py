import pandas as pd
import os
from mlproject import logger
import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from mlproject.entities.config_entity import ModelTrainerConfig



class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        # Validate file paths
        if not os.path.exists(self.config.train_data_path):
            raise FileNotFoundError(f"Training data file not found at {self.config.train_data_path}")
        if not os.path.exists(self.config.test_data_path):
            raise FileNotFoundError(f"Testing data file not found at {self.config.test_data_path}")

        # Load the data
        train_data = np.load(self.config.train_data_path, allow_pickle=True)
        test_data = np.load(self.config.test_data_path, allow_pickle=True)

        logger.info(f"Loaded train data: type={type(train_data)}, shape={train_data.shape}")
        logger.info(f"Loaded test data: type={type(test_data)}, shape={test_data.shape}")

        # Split features and target
        train_x = train_data[:, :-1]  # All columns except the last one
        train_y = train_data[:, -1]   # Only the last column
        test_x = test_data[:, :-1]    # All columns except the last one
        test_y = test_data[:, -1]     # Only the last column

        logger.info(f"Training data shape: X={train_x.shape}, y={train_y.shape}")
        logger.info(f"Testing data shape: X={test_x.shape}, y={test_y.shape}")

        # Train the model
        logger.info("Initializing RandomForestClassifier...")
        classifier = RandomForestClassifier(n_estimators=self.config.n_estimators,
                                            min_samples_split=self.config.min_samples_split,
                                            min_samples_leaf=self.config.min_samples_leaf,
                                            max_samples=self.config.max_samples,
                                            max_features=self.config.max_features,
                                            max_depth=self.config.max_depth,
                                            criterion=self.config.criterion,
                                            bootstrap=self.config.bootstrap,
                                            random_state=42)
        classifier.fit(train_x, train_y)

        logger.info("Training the model...")

        # Save the trained model
        model_path = os.path.join(self.config.root_dir, self.config.model_name)
        joblib.dump(classifier, model_path)
        logger.info(f"Model saved successfully at {model_path}")

