import keras
from keras.datasets import cifar10
import numpy as np
from keras import backend as K

def load_data():
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)

def split_train(x_train, y_train):
    n = x_train.shape[0]
    n_trains = n - 5000
    x_tune = x_train[-5000:, :]
    y_tune = y_train[-5000:, :]
    x_train = x_train[:n_trains, :]
    y_train = y_train[:n_trains, :]
    return (x_train, y_train), (x_tune, y_tune)

def preprocess(x_train, y_train, x_test, y_test, substract_pixel_mean=False):
    num_classes = 10
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    # preprocess data
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    if substract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean
    return (x_train, y_train), (x_test, y_test)

if __name__ == "__main__":
    (x_train, y_train), (x_test, y_test) = load_cifar10()
    # (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    (x_train, y_train), (x_tune, y_tune) = split_train(x_train, y_train)

