from flask import Flask, request, render_template, jsonify
import pandas as pd
from mlproject.pipeline.pipelineprediction import ChurnPredictionPipeline

app = Flask(__name__)

# Initialize the prediction pipeline
pipeline = ChurnPredictionPipeline()

@app.route('/')
def home():
    """
    Render the homepage with the input form.
    """
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle POST requests to predict churn based on user input.
    """
    try:
        # Extract input data from the form
        input_data = {
            'gender': [request.form['gender']],
            'SeniorCitizen': [int(request.form['SeniorCitizen'])],
            'Partner': [request.form['Partner']],
            'Dependents': [request.form['Dependents']],
            'tenure': [int(request.form['tenure'])],
            'PhoneService': [request.form['PhoneService']],
            'MultipleLines': [request.form['MultipleLines']],
            'InternetService': [request.form['InternetService']],
            'OnlineSecurity': [request.form['OnlineSecurity']],
            'OnlineBackup': [request.form['OnlineBackup']],
            'DeviceProtection': [request.form['DeviceProtection']],
            'TechSupport': [request.form['TechSupport']],
            'StreamingTV': [request.form['StreamingTV']],
            'StreamingMovies': [request.form['StreamingMovies']],
            'Contract': [request.form['Contract']],
            'PaperlessBilling': [request.form['PaperlessBilling']],
            'PaymentMethod': [request.form['PaymentMethod']],
            'MonthlyCharges': [float(request.form['MonthlyCharges'])],
            'TotalCharges': [float(request.form['TotalCharges'])]
        }

        # Convert input data to a DataFrame
        input_df = pd.DataFrame(input_data)

        # Make prediction using the pipeline
        prediction = pipeline.predict(input_df)

        # Return the prediction result
        return render_template('results.html', prediction=prediction)

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)