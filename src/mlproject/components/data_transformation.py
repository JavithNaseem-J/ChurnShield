import os
from mlproject import logger
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import numpy as np
import pandas as pd
from src.mlproject.entities.config_entity import DataTransformationConfig


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.label_encoders = {}
        
        # Define column types
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.cat_cols_le = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'PaperlessBilling', 'InternetService',
            'Contract', 'PaymentMethod', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        self.cols_to_drop = ['customerID']

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        try:
            data = data.copy()
            data = data.drop(columns=self.cols_to_drop, errors='ignore')
            
            # Handle TotalCharges column
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].str.strip(), errors='coerce')
            data[self.num_cols] = data[self.num_cols].fillna(data[self.num_cols].mean())
            
            # Label encode categorical columns
            for column in self.cat_cols_le:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le
            
            # Encode target column if categorical
            if self.config.target_column in data.columns and data[self.config.target_column].dtype == 'object':
                le = LabelEncoder()
                data[self.config.target_column] = le.fit_transform(data[self.config.target_column])
                self.label_encoders[self.config.target_column] = le
            
            # Save label encoders
            os.makedirs(os.path.dirname(self.config.label_encoder), exist_ok=True)
            joblib.dump(self.label_encoders, self.config.label_encoder)
            
            return data
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            raise e

    def train_test_spliting(self) -> tuple:
        try:
            data = pd.read_csv(self.config.data_path)
            data = self.preprocess_data(data)
            
            X = data.drop(columns=[self.config.target_column])
            y = data[self.config.target_column]
            
            # Apply SMOTE
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_data = X_resampled.copy()
            resampled_data[self.config.target_column] = y_resampled
            
            # Train-test split
            train, test = train_test_split(resampled_data, test_size=0.25, random_state=42)
            
            # Save splits
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            
            logger.info(f"Original data shape: {data.shape}")
            logger.info(f"Resampled data shape: {resampled_data.shape}")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")
            
            return train, test
            
        except Exception as e:
            logger.error(f"Error in train_test_spliting: {str(e)}")
            raise e

    def preprocess_features(self, train: pd.DataFrame, test: pd.DataFrame) -> tuple:
        try:
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.num_cols)
                ],
                remainder='passthrough'
            )

            # Split features and target
            train_x = train.drop(columns=[self.config.target_column])
            test_x = test.drop(columns=[self.config.target_column])
            train_y = train[self.config.target_column]
            test_y = test[self.config.target_column]

            # Apply preprocessing
            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            # Reshape targets
            train_y = train_y.values.reshape(-1, 1)
            test_y = test_y.values.reshape(-1, 1)

            # Combine processed features with targets
            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            # Save preprocessor and processed data
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            joblib.dump(preprocessor, self.config.preprocessor_path)
            
            np.save(os.path.join(self.config.root_dir, "train_processed.npy"), train_combined)
            np.save(os.path.join(self.config.root_dir, "test_processed.npy"), test_combined)

            logger.info(f"Preprocessor saved at: {self.config.preprocessor_path}")
            logger.info(f"Training data shape: {train_processed.shape}")
            logger.info(f"Testing data shape: {test_processed.shape}")
            
            return train_processed, test_processed
            
        except Exception as e:
            logger.error(f"Error in preprocess_features: {str(e)}")
            raise e
        