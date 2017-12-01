from keras.models import load_model
from resnet import *

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
    (x_train, y_train), (x_test, y_test), input_shape = prepare_data_for_resnet(substract_pixel_mean)
    filename = "cifar10_resnet_model.01.h5"
    model = load_model(filename)


