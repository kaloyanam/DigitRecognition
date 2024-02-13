import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
from load_tests import *
import streamlit as st


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Load data from dataset
def create_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.reshape(-1, 28 * 28) / 255.0
    x_test = x_test.reshape(-1, 28 * 28) / 255.0

    # Create a model with Sequential API
    model = keras.Sequential()
    model.add(keras.Input(28 * 28))
    model.add(layers.Dense(512, 'relu'))
    model.add(layers.Dense(256, 'relu'))
    model.add(layers.Dense(128,'relu'))
    model.add(layers.Dropout(0.1))
    model.add(layers.Dense(10))

    # Compile model
    model.compile(
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer = 'adam',
        metrics=['accuracy']
    )
    return [model,x_train,y_train]

# Train
def train(_model,x_train,y_train,trained):
    cp_path = os.path.dirname(os.path.abspath(__file__)) + '\\callbacks\\cp.ckpt'

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=cp_path, save_weights_only=True, verbose=1)

    if not trained:
        _model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=2, callbacks=[cp_callback])
        trained = True
    else:
        _model.load_weights(cp_path)            
    return _model


# Predict data
def predict_image(_image,_model,x_train,y_train):
    #_model.evaluate(x_train, y_train, batch_size=32, verbose=2)
    predict = _model.predict(_image)
    print(str(predict))
    print(str(_image))
    for i in predict:
        print('I guess: ' + str(np.argmax(i)))
        print('Prediction: ' + str(i))
        arr = []
        for j in i:
            arr.append(j)
        return arr