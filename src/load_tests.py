import tensorflow as tf
import cv2
import os

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

# Load images from the \images directory
def load_images(): 
    images = tf.zeros((0, 28 * 28), dtype=tf.float32)
    for i in range(10):
        image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '\\images\\' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        image = image / 255.0

        image = tf.reshape(image, shape=(1, 784))
        image = tf.cast(image, tf.float32)
        images = tf.concat([images, image], axis=0)
    return images

# Load a single image
def load_image(image):
    image = cv2.resize(image,(28,28))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = image / 255.0
    image = tf.reshape(image, shape=(1, 784))
    image = tf.cast(image, tf.float32)
    return image