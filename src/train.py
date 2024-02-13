from load_tests import *
from model import *

# For training the model
model, x_train, y_train = create_model()
model = train(model, x_train, y_train, False)
