import keras
#from keras.layers import Dense, Conv2D, BatchNormalization, Activation
#from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
from keras.models import load_model

from snapshot import SnapshotCallbackBuilder
from resnet_helper import lr_schedule,resnet_v2,resnet_v1,resnet_block
import numpy as np
import os
import sys

def load_data():
    # The data, shuffled and split between train and test sets:
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    return (x_train, y_train), (x_test, y_test)


# def preprocess(x_train, y_train, x_test, y_test, substract_pixel_mean=False):

def preprocess(x_train, y_train, x_test, y_test, substract_pixel_mean):
    num_classes = 10

    # We assume data format "channels_last".
    img_rows = x_train.shape[1]
    img_cols = x_train.shape[2]
    channels = x_train.shape[3]

    if K.image_data_format() == 'channels_first':
        # this branch is useless
        print("channels_first")
        img_rows = x_train.shape[2]
        img_cols = x_train.shape[3]
        channels = x_train.shape[1]
        x_train = x_train.reshape(x_train.shape[0], channels, img_rows, img_cols)
        x_test = x_test.reshape(x_test.shape[0], channels, img_rows, img_cols)
        input_shape = (channels, img_rows, img_cols)
    else:
        img_rows = x_train.shape[1]
        img_cols = x_train.shape[2]
        channels = x_train.shape[3]
        x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channels)
        x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channels)
        input_shape = (img_rows, img_cols, channels)

    # Normalize data.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # If subtract pixel mean is enabled
    if substract_pixel_mean:
        x_train_mean = np.mean(x_train, axis=0)
        x_train -= x_train_mean
        x_test -= x_train_mean

    print('x_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print('y_train shape:', y_train.shape)

    # Convert class vectors to binary class matrices.
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train), (x_test, y_test), input_shape

# build resnet model
def build_resnet(x_train, y_train, x_test, y_test, input_shape, batch_size, epochs, num_classes, n, version, data_augmentation):
    # prepare variables for snapshots ensemble
    # T = epochs
    # M = 3
    # alpha_zero = 0.001
    # snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

    # Computed depth from supplied model parameter n
    depth = n * 6 + 2

    # Model name, depth and version
    model_type = 'ResNet%d v%d' % (depth, version)

    if version == 2:
        model = resnet_v2(input_shape=input_shape, depth=depth)
    else:
        model = resnet_v1(input_shape=input_shape, depth=depth)

    model.compile(loss='categorical_crossentropy',
            optimizer=Adam(lr=lr_schedule(0)),
            metrics=['accuracy'])
    # model.summary()
    # print(model_type)

    # Prepare model model saving directory.
    save_dir = os.path.join(os.getcwd(), 'saved_models')
    model_name = 'cifar10_resnet_model.{epoch:02d}.h5'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join(save_dir, model_name)
    # filepath = model_type + "-{epoch:02d}-{val_acc:.2f}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

    # Prepare callbacks for model saving and for learning rate adjustment.
    checkpoint = ModelCheckpoint(filepath=filepath,
                                  monitor='val_acc',
                                  verbose=1,
                                  save_best_only=True)
    lr_scheduler = LearningRateScheduler(lr_schedule)

    lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1),
            cooldown=0,
            patience=5,
            min_lr=0.5e-6)

    callbacks = [checkpoint, lr_reducer, lr_scheduler]

    # callbacks  = snapshot.get_callbacks(model_prefix="ResNet-snap-") # try snapshot callback

    # Run training, with or without data augmentation.
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True,
                callbacks=callbacks)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                # set input mean to 0 over the dataset
                featurewise_center=False,
                # set each sample mean to 0
                samplewise_center=False,
                # divide inputs by std of dataset
                featurewise_std_normalization=False,
                # divide each input by its std
                samplewise_std_normalization=False,
                # apply ZCA whitening
                zca_whitening=False,
                # randomly rotate images in the range (deg 0 to 180)
                rotation_range=0,
                # randomly shift images horizontally
                width_shift_range=0.1,
                # randomly shift images vertically
                height_shift_range=0.1,
                # randomly flip images
                horizontal_flip=True,
                # randomly flip images
                vertical_flip=False)

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # Fit the model on the batches generated by datagen.flow().
        steps_per_epoch = int(np.ceil(x_train.shape[0] / float(batch_size)))
        history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                steps_per_epoch=steps_per_epoch,
                validation_data=(x_test, y_test),
                epochs=epochs, verbose=1, workers=4,
                callbacks=callbacks)

    '''
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    # model.predict_proba(x_test).shape
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    '''
    return model, history

def load_saved_resnet_model(filename):
    model = load_model(filename)
    return model

def evaluate_resnet_model(model, x_test, y_test, verbose=1):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    # model.predict_proba(x_test).shape
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])

def evaluate(model, x_test, y_test):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', scores[0])
    print('Test accuracy:', scores[1])
    return scores

def predict(model, x_test):
    test_classes = model.predict(x_test, verbose=0)
    test_classes = np.argmax(test_classes, axis=1)
    # print(test_classes.shape)
    return test_classes



# Model parameter
# ----------------------------------------------------------------------------
#           |      | 200-epoch | Orig Paper| 200-epoch | Orig Paper| sec/epoch
# Model     |  n   | ResNet v1 | ResNet v1 | ResNet v2 | ResNet v2 | GTX1080Ti (GTX960)
#           |      | %Accuracy | %Accuracy | %Accuracy | %Accuracy | v1 (v2) v1 (v2)
# ----------------------------------------------------------------------------
# ResNet20  |  3   | 92.16     | 91.25     | -----     | NA        | 35       72 (105)
# ResNet32  |  5   | 92.46     | 92.49     | -----     | NA        | 50
# ResNet44  |  7   | 92.50     | 92.83     | -----     | NA        | 70
# ResNet56  |  9   | 92.71     | 93.03     | 92.60     | NA        | 90 (100)
# ResNet110 |  18  | 92.65     | 93.39     | 93.03     | 93.63     | 165(180)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Training parameters
    batch_size = 64
    epochs = 2
    data_augmentation = True
    num_classes = 10

    # Subtracting pixel mean improves accuracy
    n = 3

    # Model version
    # Orig paper: version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
    version = 1
    substract_pixel_mean = True
    (x_train, y_train), (x_test, y_test), input_shape = preprocess(substract_pixel_mean)
    # model = build_resnet(x_train, y_train, x_test, y_test, input_shape, batch_size, epochs, num_classes, n, version, data_augmentation)
    # filename = "saved_models/cifar10_resnet_model.01.h5"
    # filename = "weights/ResNet-snap--6.h5"
    # model = load_saved_resnet_model(filename)
    # evaluate_resnet_model(model, x_test, y_test)
