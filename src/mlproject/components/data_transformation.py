import os
from src.mlproject import logger
from sklearn.model_selection import train_test_split,GridSearchCV,StratifiedKFold
from sklearn.preprocessing import LabelEncoder,StandardScaler,OneHotEncoder
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
        self.label_encoders = {}
        
        # Define column types
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        
        self.cat_cols_ohe = [
            'InternetService',
            'Contract',
            'PaymentMethod',
            'MultipleLines',
            'OnlineSecurity',
            'OnlineBackup',
            'DeviceProtection',
            'TechSupport',
            'StreamingTV',
            'StreamingMovies'
        ]
        
        self.cat_cols_le = [
            'gender',
            'SeniorCitizen',
            'Partner',
            'Dependents',
            'PhoneService',
            'PaperlessBilling'
        ]
        
        self.cols_to_drop = ['customerID']

    def preprocess_data(self, data):
        try:
            data = data.copy()
            data = data.drop(columns=self.cols_to_drop, errors='ignore')
            
            # Handle TotalCharges column
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].str.strip(), errors='coerce')
            data[self.num_cols] = data[self.num_cols].fillna(data[self.num_cols].mean())
            
            # Label encode all categorical columns for SMOTE
            categorical_columns = self.cat_cols_le + self.cat_cols_ohe
            for column in categorical_columns:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le
            
            # Ensure the target column is also encoded if it's categorical
            if self.config.target_column in data.columns and data[self.config.target_column].dtype == 'object':
                le = LabelEncoder()
                data[self.config.target_column] = le.fit_transform(data[self.config.target_column])
                self.label_encoders[self.config.target_column] = le
            
            return data
            
        except Exception as e:
            logger.error(f"Error in preprocess_data: {str(e)}")
            raise e

    def train_test_spliting(self):
        try:
            # Load the data
            data = pd.read_csv(self.config.data_path)
            
            # Preprocess the data first (handles categorical encoding)
            data = self.preprocess_data(data)
            
            # Separate features and target before splitting
            X = data.drop(columns=[self.config.target_column])
            y = data[self.config.target_column]
            
            # Now apply SMOTE on numeric data
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
            
        except Exception as e:
            logger.error(f"Error in train_test_spliting: {str(e)}")
            raise e

    def preprocess_features(self, train, test):
        try:
            # Identify numerical columns
            numerical_columns = train.select_dtypes(include=["int64", "float64"]).columns

            # Exclude the target column from numerical columns
            if self.config.target_column in numerical_columns:
                numerical_columns = numerical_columns.drop(self.config.target_column)

            logger.info(f"Numerical columns: {list(numerical_columns)}")
            
            
            numeric_transformer = Pipeline(steps=[
                ('scaler', StandardScaler())
            ])
            
            categorical_transformer = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False))
            ])
            
            preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, self.num_cols),
                    ('cat', categorical_transformer, self.cat_cols_ohe)
                ],
                remainder='passthrough'
            )

            # Separate features and target
            train_x = train.drop(columns=[self.config.target_column])
            test_x = test.drop(columns=[self.config.target_column])
            train_y = train[self.config.target_column]
            test_y = test[self.config.target_column]

            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            # Ensure target is 2D array
            train_y = train_y.values.reshape(-1, 1)
            test_y = test_y.values.reshape(-1, 1)

            # Combine processed features with target
            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            onehot_features = []
            for cat_col in self.cat_cols_ohe:
                unique_values = sorted(set(train_x[cat_col].astype(str)))[1:]
                onehot_features.extend([f"{cat_col}_{val}" for val in unique_values])
            
            feature_names = self.num_cols + onehot_features + self.cat_cols_le
            
            joblib.dump(preprocessor, self.config.preprocessor_path)
            joblib.dump(feature_names, 
                       os.path.join(self.config.root_dir, "feature_names.pkl"))
            
            train_data = np.column_stack((train_processed, train_y))
            test_data = np.column_stack((test_processed, test_y))
            
            np.save(os.path.join(self.config.root_dir, "train_processed.npy"), 
                   train_combined)
            np.save(os.path.join(self.config.root_dir, "test_processed.npy"), 
                   test_combined)

            logger.info("Data transformation completed successfully")
            logger.info(f"Preprocessor saved at: {self.config.preprocessor_path}")
            logger.info(f"Training data shape: {train_processed.shape}")
            logger.info(f"Testing data shape: {test_processed.shape}")
            
            return train_processed, test_processed
            
        except Exception as e:
            logger.error(f"Error in preprocess_features: {str(e)}")
            raise e
        