from flask import Flask, render_template, request
import os
import pandas as pd
import numpy as np
from mlproject.pipeline.pipelineprediction import PipelinePrediction

app = Flask(__name__)

@app.route('/')
def home():
    return "<p> Welcome to the home page </p>"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
