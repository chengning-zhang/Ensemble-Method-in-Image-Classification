import keras
from keras.models import load_model
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
model = load_model("keras_cifar10_trained_model_1.h5")
# model = load_model("saved_models/keras_cifar10_trained_model.h5")
# model = load_model("callback-save-90-0.79.hdf5")
# model = load_model("../cifar10_resnet/ResNet110-v1-04-0.64.hdf5")
# model = load_model("../cifar10_resnet/cifar10_resnet_model.04.h5-04-0.72.hdf5")
# model = load_model("../keras/examples/saved_models/cifar10_resnet_model.02.h5")

# The data, shuffled and split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
num_classes = 10
y_test_old = y_test[:]

# Convert class vectors to binary class matrices.
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
#probs = model.predict_proba(x_test)
