from flask import Flask, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import pickle
from flasgger import Swagger
import os

app = Flask(__name__)
# CORS(app)
Swagger(app)
file = open('classifier.pkl', 'rb')
model = pickle.load(file)

@app.route('/')
def welcome():
    return "Welcome all"

@app.route('/predict', methods=['Get'])
def predict_note_auth():

    """Let's authenticate the bank notes
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values

    """

    var = request.args.get('variance')
    skew = request.args.get('skewness')
    kurtosis = request.args.get('curtosis')
    entropy = request.args.get('entropy')
    prediction = model.predict([[var, skew, kurtosis, entropy]])
    return "The predicted value is : " + str(prediction)


@app.route('/predict_file', methods=['POST'])
def predict_note_test():
    """Let's authenticate the bank notes
        ---
        parameters:
            - name: file
              in: formData
              type: file
              required: true

        responses:
            200:
                description: The output values

        """
    df_test = pd.read_csv(request.files.get("file"))
    prediction = model.predict(df_test)
    return str(list(prediction))


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)