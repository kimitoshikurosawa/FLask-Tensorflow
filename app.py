#Stage 1: Importation des dependances

import os
import requests
import numpy as np
import tensorflow as tf

from scipy.misc import imread, imsave
from tensorflow.keras.datasets import fashion_mnist
from flask import Flask, request, jsonify

print(tf.__version__)

#Stage 2: chargement du modèle
with open('fashion_model_flask.json', 'r') as f:
    model_json = f.read()

model = tf.keras.models.model_from_json(model_json)
# chargement du poids du modèle
model.load_weights("fashion_model_flask.h5")

#Stage 3: Creation de l'api Flask
#Demarrage de l'api
app = Flask(__name__)

#Defining the classify_image function
@app.route("/api/v1/<string:img_name>", methods=["POST"])
def classify_image(img_name):
    #on definit le repertoire pour les upload
    upload_dir = "uploads/"
    #chargement de l'image en grayscale
    image = imread(upload_dir + img_name)

    #Definission des classes
    classes = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    #application de la prediction avec le modèle prédfini
    prediction = model.predict([image.reshape(1, 28*28)])

    #on retourne le resutlat à l'utilisateur
    return jsonify({"object_identified":classes[np.argmax(prediction[0])]})

#demarrage de l'appliation HTTP
app.run(port=5000, debug=False)