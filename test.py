import matplotlib.pyplot as plt
import json
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("recognizer")

with open("new") as file:
    image = json.load(file)
   #  plt.imshow(image)
   #  plt.show()


image = (np.array(image) / 255.0).reshape(1, 28, 28, 1)
plt.imshow(image.reshape(28, 28, 1))
plt.show()


def predictNumber(image, model):
    predictions = model.predict(image, verbose=False)
    max = predictions.max()
    for i in range(predictions.size):
        if predictions[0][i] == max:
            return i


print(predictNumber(image, model))
