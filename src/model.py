from tensorflow import keras
from keras import layers
from keras.datasets import mnist
import numpy as np
from load_tests import load_images

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Load data from dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.reshape(-1, 28 * 28) / 255.0
x_test = x_test.reshape(-1, 28 * 28) / 255.0

# Create a model with Sequential API
model = keras.Sequential()
model.add(keras.Input(28 * 28))
model.add(layers.Dense(512, 'relu'))
model.add(layers.Dense(256, 'relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10))

# Compile model
model.compile(
    loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = 'adam',
    metrics=['accuracy']
)

# Train 
# TODO - save data (checkpoints)
model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=2)
model.evaluate(x_test, y_test, batch_size=32, verbose=2)


# Predict data
predict = model.predict(load_images())
for i in range(10):
    print('I guess: ' + str(np.argmax(predict[i])))
    print('Prediction: ' + str(predict[i]))