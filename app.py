from flask import Flask, render_template, request
import sys
import os
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.append(str(src_path))

from src.mlproject.pipeline.pipelineprediction import ChurnPredictionPipeline
import pandas as pd

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data - matching the camelCase naming in your HTML form
        data = {
            'gender': [request.form['gender']],
            'SeniorCitizen': [int(request.form['seniorCitizen'])],
            'Partner': [request.form['partner']],
            'Dependents': [request.form['dependents']],
            'tenure': [float(request.form['tenure'])],
            'PhoneService': [request.form['phoneService']],
            'MultipleLines': [request.form['multipleLines']],
            'InternetService': [request.form['internetService']],
            'OnlineSecurity': [request.form['onlineSecurity']],
            'OnlineBackup': [request.form['onlineBackup']],
            'DeviceProtection': [request.form['deviceProtection']],
            'TechSupport': [request.form['techSupport']],
            'StreamingTV': [request.form['streamingTV']],
            'StreamingMovies': [request.form['streamingMovies']],
            'Contract': [request.form['contract']],
            'PaperlessBilling': [request.form['paperlessBilling']],
            'PaymentMethod': [request.form['paymentMethod']],
            'MonthlyCharges': [float(request.form['monthlyCharges'])],
            'TotalCharges': [float(request.form['totalCharges'])]
        }
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Initialize pipeline and make prediction
        pipeline = ChurnPredictionPipeline()
        result = pipeline.predict(df)
        
        return render_template('results.html', 
                             prediction=result['churn_status'],
                             probability=round(result['churn_probability'] * 100, 2))
    
    except Exception as e:
        return render_template('results.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)