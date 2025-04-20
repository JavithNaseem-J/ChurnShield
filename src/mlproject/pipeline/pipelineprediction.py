import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from mlproject.utils.common import read_yaml
from mlproject.constants import *


class ChurnPredictionPipeline:
    def __init__(self):
        self.schema = read_yaml(Path('schema.yaml'))
        self.preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
        self.model_path = Path('artifacts/model_trainer/model.joblib')
        self.label_encoders_path = Path('artifacts/data_transformation/label_encoders.pkl')
        
        self.num_cols = self.schema.num_cols
        self.cat_cols = self.schema.cat_cols
        self.target_column = self.schema.TARGET_COLUMN.name
    

        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.label_encoders_path):
            raise FileNotFoundError(f"Label encoders file not found: {self.label_encoders_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path) 

    def preprocess_input(self, input_data):
        
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        data = input_data.copy()
        
        for column in self.cat_cols:
            if column in data.columns and column in self.label_encoders:
                if column != 'SeniorCitizen':
                    data[column] = self.label_encoders[column].transform(data[column].astype(str))
        
        for column in self.num_cols:
            if column in data.columns:
                data[column] = data[column].astype(float)
        
        try:
            processed_data = self.preprocessor.transform(data)
            return processed_data
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise

    def predict(self, input_data):
        
        try:
            processed_data = self.preprocess_input(input_data)
            
            
            prediction_result = int(self.model.predict(processed_data)[0])
            probabilities = self.model.predict_proba(processed_data)[0]
            
            # Get the class probabilities
            positive_class_index = 1 if len(probabilities) > 1 else 0
            churn_probability = probabilities[positive_class_index]
            
            # Decode the prediction using the label encoder if available
            if self.target_column in self.label_encoders:
                churn_status = self.label_encoders[self.target_column].inverse_transform([prediction_result])[0]
            else:
                churn_status = "Yes" if prediction_result == 1 else "No"
            
            return {
                "churn_status": churn_status,
                "churn_probability": float(churn_probability)
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise