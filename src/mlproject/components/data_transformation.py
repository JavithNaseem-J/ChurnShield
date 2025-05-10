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
        self.num_cols = config.num_cols
        self.cat_cols_le =  config.cat_cols
        self.cols_to_drop = config.columns_to_drop

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
    
            data = data.copy()
            data = data.drop(columns=self.cols_to_drop, errors='ignore')
            
            for col in self.num_cols:
                if col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    data[col] = data[col].fillna(data[col].median())
            
            # Label encode categorical columns
            for column in self.cat_cols_le:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le
            
            # Encode target column if categorical
            if self.config.target_column in data.columns:
                le = LabelEncoder()
                data[self.config.target_column] = le.fit_transform(data[self.config.target_column])
                self.label_encoders[self.config.target_column] = le
            
            # Save label encoders
            os.makedirs(os.path.dirname(self.config.label_encoder), exist_ok=True)
            joblib.dump(self.label_encoders, self.config.label_encoder)
            
            return data
            

    def train_test_spliting(self) -> tuple:
            data = pd.read_csv(self.config.data_path)
            data = self.preprocess_data(data)
            
            X = data.drop(columns=[self.config.target_column])
            y = data[self.config.target_column]
            
            smote = SMOTE(random_state=42)
            X_resampled, y_resampled = smote.fit_resample(X, y)
            
            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_data = X_resampled.copy()
            resampled_data[self.config.target_column] = y_resampled
            
            train, test = train_test_split(resampled_data, test_size=0.25, random_state=42)
            
            train_path = os.path.join(self.config.root_dir, "train.csv")
            test_path = os.path.join(self.config.root_dir, "test.csv")
            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)
            
            logger.info(f"Original data shape: {data.shape}")
            logger.info(f"Resampled data shape: {resampled_data.shape}")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")
            
            return train, test
            
    def preprocess_features(self, train, test) -> tuple:

        logger.info(f"Numeric columns for transformation: {self.num_cols}")
            
        valid_num_cols = [col for col in self.num_cols if col in train.columns]

        if not valid_num_cols:
            raise ValueError(f"None of the specified numeric columns {self.num_cols} exist in the data")
            
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
            
        preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, valid_num_cols)
                ],
                remainder='passthrough'
        )

        # Split features and target
        train_x = train.drop(columns=[self.config.target_column])
        test_x = test.drop(columns=[self.config.target_column])
        train_y = train[self.config.target_column]
        test_y = test[self.config.target_column]

        train_processed = preprocessor.fit_transform(train_x)
        test_processed = preprocessor.transform(test_x)

        # Reshape 
        train_y = train_y.values.reshape(-1, 1)
        test_y = test_y.values.reshape(-1, 1)

        train_combined = np.hstack((train_processed, train_y))
        test_combined = np.hstack((test_processed, test_y))

        os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
        joblib.dump(preprocessor, self.config.preprocessor_path)
            
        np.save(os.path.join(self.config.root_dir, "train_processed.npy"), train_combined)
        np.save(os.path.join(self.config.root_dir, "test_processed.npy"), test_combined)

        logger.info(f"Preprocessor saved at: {self.config.preprocessor_path}")
        logger.info(f"Training data shape: {train_processed.shape}")
        logger.info(f"Testing data shape: {test_processed.shape}")
            
        return train_processed, test_processed
            