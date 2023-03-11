from flask import Flask, request, send_from_directory
from flask_cors import CORS, cross_origin
import numpy as np
import cv2
import base64
import json
import joblib
import string


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

letters = string.ascii_uppercase


model = joblib.load("A_Z_model.sav")


def predictLetter(image, model):
    predictions = model.predict(image).argmax()
    return letters[predictions]


@app.route('/<path:path>')
def serve_files(path):
    return send_from_directory("client", path)


@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    print("connected")
    image = str(request.data).split(",")[1]
    nparr = np.frombuffer(base64.b64decode(image), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (28, 28))
    # cv2.imwrite("image.png", img)
    nparr = np.array(img)
    nparr = (nparr.__invert__() / 255.0).reshape(1, 28, 28, 1)
    print(nparr)
    return predictLetter(nparr, model)


app.run("0.0.0.0", 3002, debug=True)
