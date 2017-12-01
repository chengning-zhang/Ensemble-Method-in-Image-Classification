import keras
from keras.datasets import mnist
import numpy as np
from keras import backend as K

def load_data():
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    img_rows, img_cols = 28, 28
    if K.image_data_format() == 'channels_first':
        x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
        input_shape = (1, img_rows, img_cols)
    else:
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
        input_shape = (img_rows, img_cols, 1)
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
    print("Hello UW!")
    (x_train, y_train), (x_test, y_test) = load_data()
    

