from flask import Flask, render_template, request
import os 
import numpy as np
import pandas as pd
from mlproject.pipeline.pipelineprediction import PredictionPipeline


app = Flask(__name__) # initializing a flask app

@app.route('/',methods=['GET']) 
def homePage():
    return render_template("index.html")


@app.route('/train',methods=['GET'])  # route to train the pipeline
def training():
    os.system("python main.py")
    return "Training Successful!" 


@app.route('/predict', methods=['POST', 'GET'])
def index():
    if request.method == 'POST':
        try:
            # Read user inputs
            fixed_acidity = float(request.form['fixed_acidity'])
            volatile_acidity = float(request.form['volatile_acidity'])
            citric_acid = float(request.form['citric_acid'])
            residual_sugar = float(request.form['residual_sugar'])
            chlorides = float(request.form['chlorides'])
            free_sulfur_dioxide = float(request.form['free_sulfur_dioxide'])
            total_sulfur_dioxide = float(request.form['total_sulfur_dioxide'])
            density = float(request.form['density'])
            pH = float(request.form['pH'])
            sulphates = float(request.form['sulphates'])
            alcohol = float(request.form['alcohol'])

            # Convert data into array for prediction
            data = np.array([fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
                             chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
                             pH, sulphates, alcohol]).reshape(1, -1)

            # Create prediction pipeline object
            obj = PredictionPipeline()

            # Predict the original label and its meaning
            predicted_quality, quality_meaning = obj.predict(data)

            return render_template('results.html', 
                                   prediction=str(predicted_quality), 
                                   meaning=quality_meaning)

        except Exception as e:
            print('The Exception message is: ', e)
            return 'Something went wrong'

    else:
        return render_template('index.html')


if __name__ == "__main__":
	app.run(host="0.0.0.0", port = 8080)