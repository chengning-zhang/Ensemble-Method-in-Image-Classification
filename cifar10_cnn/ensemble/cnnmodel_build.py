from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras
import numpy as np
from snapshot import SnapshotCallbackBuilder

def build_model(x_train, num_classes):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    # initiate RMSprop optimizer
    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy'])
    return model

def train_snapshot(x_train, y_train, x_test, y_test, model, batch_size, epochs, M, alpha_zero, data_augmentation=True):
    # prepare variables for snapshots ensemble
    T = epochs
    # M = 3
    # alpha_zero = 0.001
    snapshot = SnapshotCallbackBuilder(T, M, alpha_zero)

    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # callback
        # filepath="saved_models/callback-save-{epoch:02d}-{val_acc:.2f}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, period=10)
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        # callbacks_list = [checkpoint]

        callbacks_list  = snapshot.get_callbacks(model_prefix="cnn-snapshot-") # add snapshot callback

        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(x_train, y_train,
            batch_size=batch_size),
            steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks_list,
            workers=4)
        return model


def train(x_train, y_train, x_test, y_test, model, batch_size, epochs, data_augmentation=True):
    if not data_augmentation:
        print('Not using data augmentation.')
        history = model.fit(x_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(x_test, y_test),
                shuffle=True)
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=True,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        # Compute quantities required for feature-wise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(x_train)

        # callback
        filepath="saved_models/callback-save-{epoch:02d}-{val_acc:.2f}.hdf5"
        # checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, period=10)
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
        callbacks_list = [checkpoint]

        # Fit the model on the batches generated by datagen.flow().
        history = model.fit_generator(datagen.flow(x_train, y_train,
            batch_size=batch_size),
            steps_per_epoch=int(np.ceil(x_train.shape[0] / float(batch_size))),
            epochs=epochs,
            validation_data=(x_test, y_test),
            # callbacks=callbacks_list,
            workers=4)
    return model, history

def evaluate(model, x_test, y_test):
    # Score trained model.
    scores = model.evaluate(x_test, y_test, verbose=1)
    # print('Test loss:', scores[0])
    # print('Test accuracy:', scores[1])
    return scores

def predict(model, x_test):
    test_classes = model.predict(x_test, verbose=0)
    test_classes = np.argmax(test_classes, axis=1)
    # print(test_classes.shape)
    return test_classes

def get_probs(model, x_test):
    return model.predict_proba(x_test)

# ntests = x_test.shape[0]
# errors = np.count_nonzero(test_classes - y_test_old.reshape((ntests,)))
# print('Test accuracy: %f %%' % ((ntests - errors)/float(ntests)*100))

if __name__ == "__main__":
    print("Hello world")
