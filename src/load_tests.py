import tensorflow as tf
import cv2
import os

#tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)

def load_images(): 
    images = tf.zeros((0, 28 * 28), dtype=tf.float32)
    for i in range(10):
        image = cv2.imread(os.path.dirname(os.path.abspath(__file__)) + '\\images\\' + str(i) + '.png', cv2.IMREAD_GRAYSCALE)
        image = image / 255.0

        image = tf.reshape(image, shape=(1, 784))
        image = tf.cast(image, tf.float32)
        images = tf.concat([images, image], axis=0)
    return images
def load_image(image):
    image = cv2.resize(image,(28,28))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # f = open('newimage.txt','w')
    # for i in image:
    #     f.write(str(i)+'\n')
    # f.close()
    image = image / 255.0
    image = tf.reshape(image, shape=(1, 784))
    image = tf.cast(image, tf.float32)
    return image