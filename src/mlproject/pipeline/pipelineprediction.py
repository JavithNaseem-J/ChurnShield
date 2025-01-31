import joblib 
import numpy as np
import pandas as pd
from pathlib import Path



class PredictionPipeline:
    def __init__(self):
        # Load trained model
        self.model = joblib.load(Path('artifacts/model_trainer/model.joblib'))

        # Load the label encoder
        self.label_encoder = joblib.load(Path('artifacts/data_transformation/label_encoder.pkl'))

        # ✅ Define quality meaning inside __init__
        self.quality_meaning = {
            3: "Poor",
            4: "Below Average",
            5: "Average",
            6: "Good",
            7: "Excellent"
        }

    def predict(self, data):
        # Predict encoded labels
        encoded_prediction = self.model.predict(data)

        # Convert prediction to int
        encoded_prediction = np.round(encoded_prediction).astype(int)

        # Convert encoded label back to original label
        original_prediction = self.label_encoder.inverse_transform(encoded_prediction)[0]

        # ✅ Get quality meaning safely
        meaning = self.quality_meaning.get(original_prediction, "Unknown")

        return original_prediction, meaning