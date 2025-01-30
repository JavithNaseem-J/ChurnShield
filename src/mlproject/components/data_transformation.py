import os
from src.mlproject import logger
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd
from mlproject.entities.config_entity import DataTransformationConfig



class DataTransformation:
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()

    def train_test_spliting(self):
        data = pd.read_csv(self.config.data_path)
        
        # Separate features and target before splitting
        X = data.drop(columns=[self.config.target_column])
        y = data[self.config.target_column]
        
        # Encode target labels
        y = self.label_encoder.fit_transform(y)
        
        # Apply SMOTE before train-test split
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        
        # Convert back to DataFrame to maintain column names
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        
        # Combine features and target for saving
        resampled_data = X_resampled.copy()
        resampled_data[self.config.target_column] = y_resampled
        
        # Perform train-test split on resampled data
        train, test = train_test_split(resampled_data, test_size=0.25, random_state=42)

        train_path = os.path.join(self.config.root_dir, "train.csv")
        test_path = os.path.join(self.config.root_dir, "test.csv")
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)

        logger.info("Applied SMOTE and split data into training and test sets")
        logger.info(f"Original data shape: {data.shape}")
        logger.info(f"Resampled data shape: {resampled_data.shape}")
        logger.info(f"Training data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")

        return train, test
    
    def preprocess_features(self, train, test):
        # Identify numerical columns
        numerical_columns = train.select_dtypes(include=["int64", "float64"]).columns

        # Exclude the target column from numerical columns
        if self.config.target_column in numerical_columns:
            numerical_columns = numerical_columns.drop(self.config.target_column)

        logger.info(f"Numerical columns: {list(numerical_columns)}")

        # Preprocessing pipelines
        num_pipeline = Pipeline(steps=[
            ("scaler", StandardScaler())
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipeline, numerical_columns),
            ],
            remainder="passthrough"
        )

        # Separate features and target
        train_x = train.drop(columns=[self.config.target_column])
        test_x = test.drop(columns=[self.config.target_column])
        train_y = train[self.config.target_column]
        test_y = test[self.config.target_column]

        # Fit preprocessor and transform features
        train_processed = preprocessor.fit_transform(train_x)
        test_processed = preprocessor.transform(test_x)

        # Ensure target is 2D array
        train_y = train_y.values.reshape(-1, 1)
        test_y = test_y.values.reshape(-1, 1)

        # Combine processed features with target
        train_combined = np.hstack((train_processed, train_y))
        test_combined = np.hstack((test_processed, test_y))

        # Save preprocessor and label encoder
        joblib.dump(preprocessor, self.config.preprocessor_path)
        label_encoder_path = os.path.join(self.config.root_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, label_encoder_path)
        
        logger.info(f"Preprocessor saved at {self.config.preprocessor_path}")
        logger.info(f"Label encoder saved at {label_encoder_path}")

        # Save processed data
        np.save(os.path.join(self.config.root_dir, "train_processed.npy"), train_combined)
        np.save(os.path.join(self.config.root_dir, "test_processed.npy"), test_combined)

        logger.info("Preprocessed train and test data saved successfully.")
        return train_processed, test_processed
        