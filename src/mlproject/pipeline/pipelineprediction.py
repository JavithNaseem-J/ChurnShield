import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

    
    def predict(self, data):
        prediction = self.model.predict(data)

        return prediction
    

class PredictionPipeline:
    def __init__(self):
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))
        self.label_encoder = joblib.load(Path('artifacts/data_transformation/label_encoder.pkl'))
        
        # Define quality meanings
        self.quality_meanings = {
            3: "Bad (Poor Quality)",
            4: "Below Average",
            5: "Average",
            6: "Good",
            7: "Excellent"
        }
    
    def predict(self, data):
        # Get encoded prediction
        predict = self.model.predict(data)
        
        # Convert encoded prediction back to original value
        original_prediction = self.label_encoder.inverse_transform(predict)[0]
        
        # Get the quality meaning
        quality_meaning = self.quality_meanings.get(original_prediction, "Unknown")
        prediction = f"Wine Quality: {original_prediction} - {quality_meaning}"
        return prediction