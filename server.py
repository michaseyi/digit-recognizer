from flask import Flask, request
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import json


model = tf.keras.models.load_model("recognizer")
app = Flask(__name__)
CORS(app)


def predictNumber(image, model):
    predictions = model.predict(image).reshape(10)
    predictions = list(zip(predictions, range(predictions.size)))
    return sorted(predictions, reverse=True)[0][1]


@app.route('/predict', methods=['POST'])
def predict():
   #  with open("new", 'w') as file:
   #      json.dump(request.json, file)
    image = np.array(request.json)
    image = (image / 255.0).reshape(1, 28, 28, 1)
    return "{}".format(predictNumber(image, model))


app.run(debug=True)
