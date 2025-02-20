import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

class PredictionPipeline:
    def __init__(self):
        """
        Initialize the PredictionPipeline by loading the preprocessor, label encoders, and the trained model.
        """
        # Define paths for the preprocessor and model
        self.preprocessor_path = Path('artifacts/data_transformation/preprocessor.pkl')
        self.model_path = Path('artifacts/model_trainer/model.joblib')
        self.label_encoders_path = Path('artifacts/data_transformation/label_encoders.pkl')
        
        # Define column types
        self.num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        self.cat_cols = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents',
            'PhoneService', 'PaperlessBilling', 'InternetService',
            'Contract', 'PaymentMethod', 'MultipleLines',
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]

        # Load the preprocessor, label encoders, and the trained model
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found: {self.preprocessor_path}")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        if not os.path.exists(self.label_encoders_path):
            raise FileNotFoundError(f"Label encoders file not found: {self.label_encoders_path}")

        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path)
        self.target_column = 'Churn'  # Assuming this is your target column

    def preprocess_input(self, input_data):
        """
        Preprocess the input data using the label encoders and preprocessor.
        Args:
            input_data (DataFrame): A single-row DataFrame containing feature values.
        Returns:
            ndarray: Preprocessed input data ready for prediction.
        """
        # Ensure the input data is in the correct format
        if not isinstance(input_data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        # Make a copy to avoid modifying the original data
        data = input_data.copy()
        
        # Handle TotalCharges column
        if 'TotalCharges' in data.columns:
            data['TotalCharges'] = pd.to_numeric(data['TotalCharges'].astype(str).str.strip(), errors='coerce')
            # Fill any missing values with mean from training
            if data['TotalCharges'].isnull().any():
                data['TotalCharges'] = data['TotalCharges'].fillna(0)
        
        # Convert SeniorCitizen to int explicitly (it's often stored as '0'/'1' strings)
        if 'SeniorCitizen' in data.columns:
            data['SeniorCitizen'] = data['SeniorCitizen'].astype(int)
        
        # Encode categorical features using saved label encoders
        for column in self.cat_cols:
            if column in data.columns and column in self.label_encoders:
                # Skip SeniorCitizen as it's already converted to int
                if column != 'SeniorCitizen':
                    data[column] = self.label_encoders[column].transform(data[column].astype(str))
        
        # Make sure numeric columns are proper floats
        for column in self.num_cols:
            if column in data.columns:
                data[column] = data[column].astype(float)
        
        # Preprocess the encoded data using the saved preprocessor
        try:
            processed_data = self.preprocessor.transform(data)
            return processed_data
        except Exception as e:
            print(f"Error during preprocessing: {str(e)}")
            raise

    def predict(self, input_data):
        """
        Preprocess input data and make predictions.
        Args:
            input_data (DataFrame): A single-row DataFrame containing feature values.
        Returns:
            dict: Dictionary containing prediction result and probability.
        """
        try:
            # Preprocess the input data
            processed_data = self.preprocess_input(input_data)
            
            # LightGBM compatibility check
            is_lgbm = 'LGBMClassifier' in str(type(self.model))
            
            if is_lgbm:
                # Get feature names from the model
                feature_names = self.model.feature_name_
                
                # Create a pandas DataFrame with the right column names
                # This is crucial for LightGBM which needs feature names to match
                processed_df = pd.DataFrame(processed_data, columns=[f'Column_{i}' for i in range(processed_data.shape[1])])
                
                # Safety check to ensure feature count matches
                if processed_df.shape[1] != len(feature_names):
                    # We need to handle this mismatch - for now use the first N columns
                    processed_df.columns = feature_names[:processed_df.shape[1]]
                    print(f"Warning: Feature count mismatch. Using first {processed_df.shape[1]} features.")
                else:
                    processed_df.columns = feature_names
                
                # Make the prediction
                prediction_result = int(self.model.predict(processed_df)[0])
                probabilities = self.model.predict_proba(processed_df)[0]
            else:
                # For other model types
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