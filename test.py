import pandas as pd
from mlproject.pipeline.pipelineprediction import ChurnPredictionPipeline

def test_prediction():
    # Create sample input data
    input_data = pd.DataFrame({
        'gender': ['Male'],
        'SeniorCitizen': [0],
        'Partner': ['Yes'],
        'Dependents': ['No'],
        'tenure': [24],
        'PhoneService': ['Yes'],
        'MultipleLines': ['No'],
        'InternetService': ['Fiber optic'],
        'OnlineSecurity': ['No'],
        'OnlineBackup': ['Yes'],
        'DeviceProtection': ['No'],
        'TechSupport': ['No'],
        'StreamingTV': ['Yes'],
        'StreamingMovies': ['No'],
        'Contract': ['Month-to-month'],
        'PaperlessBilling': ['Yes'],
        'PaymentMethod': ['Electronic check'],
        'MonthlyCharges': [74.40],
        'TotalCharges': [1763.75]
    })
    
    # Create an instance of the prediction pipeline
    pipeline = ChurnPredictionPipeline()
    
    # Make a prediction
    try:
        result = pipeline.predict(input_data)
        print("\n=== PREDICTION RESULT ===")
        print(f"Prediction: {result['churn_status']}")
        print(f"Probability: {result['churn_probability']:.2f}")
    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_prediction()