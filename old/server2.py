from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json
import joblib
import string

letters = string.ascii_lowercase


model = joblib.load("A_Z_model.sav")

app = Flask(__name__)
CORS(app)


def predictLetter(image, model):
    predictions = model.predict(image).argmax()
    return letters[predictions]


@app.route("/", methods=['GET'])
def home():
    with open("index.html") as file:
        return file.read()


@app.route("/styles.css")
def get_route():
    with open("styles.css") as file:
        return file.read()


@app.route('/predict', methods=['POST'])
def predict():
    image = np.array(request.json)
    image = (image / 255.0).reshape(1, 28, 28, 1)
    return predictLetter(image, model)


app.run("0.0.0.0", 3002, debug=True)
