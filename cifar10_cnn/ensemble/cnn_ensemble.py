from cnnmodel_build import *
from data_preparation import *
from keras.models import load_model
import numpy as np
import sys
import os

from keras.models import Sequential
from keras.layers import Dense

def mostcommon(array):
    '''return the most common value of an array'''
    return np.bincount(array).argmax()

def weighted_vote(x_test, models, accuracy_records, num_classes=10):
    ''' return final_predict based on weighted_vote of all the learners in models
        weight is the the accuracy of each learner
    '''
    n_learners = len(models)
    n_tests = x_test.shape[0]
    # final_predict = np.zeros((n_tests, 1), dtype="int64")
    probs = np.zeros((n_tests, num_classes))
    for i in range(n_learners):
        accuracy = accuracy_records[i]
        model = models[i]
        probs = probs + accuracy*model.predict_proba(x_test)
    return np.argmax(probs, axis=1)

def majority_vote(x_test, models, accuracy_records):
    ''' return final_predict based on majority vote of all the learners in models
    '''
    n_learners = len(models)
    n_tests = x_test.shape[0]
    predictions = np.zeros((n_tests, n_learners), dtype="int64")
    for i in range(n_learners):
        model = models[i]
        predictions[:, i] = predict(model, x_test) # each column stores one learner's prediction
    final_predict = np.zeros((n_tests, 1), dtype="int64")
    for i in range(n_tests):
        final_predict[i] = mostcommon(predictions[i, :])
    return final_predict

def cross_validation():
    '''use different parameters and pick the best one to be the final learner'''
    pass

def adaboost(n_learners, epochs_lst, batch_size, sample_ratio=3, filename="temp.txt", file_prefix=""):
    ''' adaboost of multi classification'''
    num_classes = 10
    K = float(num_classes)
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_test_old = y_test[:] # save for error calculation
    y_train_old = y_train[:]
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    # weights = 1.0/n_trains*np.ones(n_trains) # initialize instance weights
    weights = [1.0/n_trains for k in range(n_trains)]
    M = sample_ratio*n_trains # >> sample a large (>> m) unweighted set of instance according to p
    test_accuracy_records = []
    alphas = []
    for i in range(n_learners):
        # weights = weights/sum(weights)
        sum_weights = sum(weights)
        weights = [weight/sum_weights for weight in weights]
        epochs = epochs_lst[i]
        model = build_model(x_train, num_classes)

        train_picks = np.random.choice(n_trains, M, weights)

        x_train_i = x_train[train_picks, :]
        y_train_i = y_train[train_picks, :]
        model, history = train(x_train_i, y_train_i, x_test, y_test, model, batch_size, epochs)
        print("model " + str(i))
        predicts = predict(model, x_train_i)
        y_ref = y_train_old[train_picks, :].reshape((M, ))
        num_error = np.count_nonzero(predicts - y_ref)
        error = float(num_error)/M
        w_changed = np.zeros(n_trains)
        alpha = np.log((1 - error)/error) + np.log(K - 1)
        for j in range(M):
            index = train_picks[j]
            if predicts[j] != y_ref[j] and w_changed[index] == 0:
                w_changed[index] = 1
                weights[index] = weights[index] * np.exp(alpha)
        alphas.append(alpha)
        print("alpha = " + str(alpha))
        models.append(model) # save base learner
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])

    final_predict = majority_vote(x_test, models, alphas)
    errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")

    print("sample ratio: %f" % (sample_ratio))
    print('ensemble test accuracy: %f' % ((n_tests - errors)/float(n_tests)))
    out_file.write('sample ratio: %f\n' % (sample_ratio))
    out_file.write('ensemble test accuracy: %f\n' % ((n_tests - errors)/float(n_tests)))

    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()
    ## check diversity
    # for i in range(n_learners):
    #    for j in range(i+1, n_learners):
    #        print("the diversity between %d and %d is %f" %(i, j, diversity(x_test, y_test, y_test_old, models[i], models[j])))

def diversity(x_data, y_data, y_data_old, model1, model2):
    scores1 = evaluate(model1, x_data, y_data)
    accuracy1 = scores1[1]
    scores2 = evaluate(model2, x_data, y_data)
    accuracy2 = scores2[1]
    predicts1 = predict(model1, x_data)
    predicts2 = predict(model2, x_data)
    n_train = x_data.shape[0]
    num_correct = 0
    for j in range(n_train):
        if predicts1[j] == y_data_old[j] or predicts2[j] == y_data_old[j]:
            num_correct += 1
    combined_acc = num_correct/n_train
    diver = combined_acc - max(accuracy1, accuracy2)
    return diver

def bagging_train_model(n_learners, epochs_lst, batch_size, votefuns, filename="temp.txt", file_prefix=""):
    '''bagging, use unique model, can use multiple vote functions, votefuns are vote
       functions list
    '''
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    train_accuracy_records = []
    test_accuracy_records = []
    for i in range(n_learners):
        epochs = epochs_lst[i]
        model = build_model(x_train, num_classes)
        train_picks = np.random.choice(n_trains, n_trains)
        x_train_i = x_train[train_picks, :]
        y_train_i = y_train[train_picks, :]
        model, history = train(x_train_i, y_train_i, x_test, y_test, model, batch_size, epochs)
        print("model %d finished" % (i))
        train_accuracy_records.append(history.history['acc'][-1])
        test_accuracy_records.append(history.history['val_acc'][-1])
        models.append(model) # save base learner

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    for votefun in votefuns:
        # get weighted vote or majority vote based on the votefun
        final_predict = votefun(x_test, models, train_accuracy_records)

        errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))
        out_file.write("votefun is\n")
        out_file.write(str(votefun) + "\n")
        out_file.write('ensemble test accuracy: %0.6f \n' % ((n_tests - errors)/float(n_tests)))
        print("votefun is ")
        print(votefun)
        print('ensemble test accuracy: %0.6f' % ((n_tests - errors)/float(n_tests)))
        for i in range(n_learners):
            print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
            out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()

def bagging_loading_model(n_learners, saved_model_files, votefuns, filename="temp.txt", file_prefix=""):
    '''load models from saved files
       votefuns are vote functions list
    '''
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    train_accuracy_records = []
    test_accuracy_records = []
    for i in range(n_learners):
        model_file = saved_model_files[i]
        model = load_model(model_file)
        print("model " + str(i))
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])
        scores = evaluate(model, x_train, y_train)
        train_accuracy_records.append(scores[1])
        models.append(model) # save base learner

    filename = file_prefix + filename
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    for votefun in votefuns:
        # get weighted vote or majority vote based on the votefun
        final_predict = votefun(x_test, models, train_accuracy_records)

        errors = np.count_nonzero(final_predict.reshape((n_tests, )) - y_test_old.reshape((n_tests,)))
        out_file.write("votefun is\n")
        out_file.write(str(votefun) + "\n")
        out_file.write('ensemble test accuracy: %0.6f \n' % ((n_tests - errors)/float(n_tests)))
        print("votefun is ")
        print(votefun)
        print('ensemble test accuracy: %0.6f' % ((n_tests - errors)/float(n_tests)))
        for i in range(n_learners):
            print("learner %d (model_file = %s): %0.6f" % (i, saved_model_files[i], test_accuracy_records[i]))
            out_file.write("learner %d (model_file = %s): %0.6f\n" % (i, saved_model_files[i], test_accuracy_records[i]))
    out_file.close()


def stack_train_model(n_learners, epochs_lst, batch_size, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    test_accuracy_records = []
    for i in range(n_learners):
        epochs = epochs_lst[i]
        model = build_model(x_train, num_classes)
        model, history = train(x_train, y_train, x_test, y_test, model, batch_size, epochs)
        print("model %d finished" % (i))
        test_accuracy_records.append(history.history['val_acc'][-1])
        models.append(model) # save base learner

    # construct meta learning problem
    meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    meta_y_train = y_train # use one hot encode
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'stacking_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (epochs = %d): %0.6f" % (i, epochs_lst[i], test_accuracy_records[i]))
        out_file.write("learner %d (epochs = %d): %0.6f\n" % (i, epochs_lst[i], test_accuracy_records[i]))
    out_file.close()


def stack_loading_model(saved_model_files, meta_epochs=40, filename="temp.txt"):
    '''stacking multiple saved models'''
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data() # cifar-10
    y_test_old = y_test[:] # save for error calculation
    y_train_old = y_train[:]
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)

    models = []
    n_trains = x_train.shape[0]
    n_tests = x_test.shape[0]
    test_accuracy_records = []
    n_learners = len(saved_model_files)
    for i in range(n_learners):
        model_file = saved_model_files[i]
        model = load_model(model_file)
        print("model " + str(i))
        scores = evaluate(model, x_test, y_test)
        test_accuracy_records.append(scores[1])
        models.append(model) # save base learner
    # construct meta learning problem
    meta_x_train = np.zeros((n_trains, n_learners*num_classes), dtype="float32")
    meta_x_test = np.zeros((n_tests, n_learners*num_classes), dtype="float32")
    for i in range(n_learners):
        meta_x_train[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_train, verbose=0)
        meta_x_test[:, i*num_classes:i*num_classes + num_classes] = models[i].predict(x_test, verbose=0)
    meta_y_train = y_train # use one hot encode
    meta_y_test = y_test
    super_model = meta_model(n_learners, num_classes)
    # callbacks
    save_dir = os.path.join(os.getcwd(), 'stacking_models')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    model_name="stack-{epoch:03d}-{val_acc:.4f}.hdf5"
    filepath = os.path.join(save_dir, model_name)
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
    callbacks_list = [checkpoint]

    super_model.fit(meta_x_train, meta_y_train, batch_size=128, epochs=meta_epochs, validation_data=(meta_x_test, meta_y_test), shuffle=True, callbacks=callbacks_list)
    scores = super_model.evaluate(meta_x_test, meta_y_test, verbose=1)
    print(filename)
    out_file = open(filename, "a")
    out_file.write("--------------------------------------------\n")
    print('Stack test accuracy: ', scores[1])
    out_file.write('Stack test accuracy: %0.6f\n' % (scores[1]))
    for i in range(n_learners):
        print("learner %d (model_file = %s): %0.6f" % (i, saved_model_files[i], test_accuracy_records[i]))
        out_file.write("learner %d (model_file = %s): %0.6f\n" % (i, saved_model_files[i], test_accuracy_records[i]))
    out_file.close()

def test1():
    n_learners = 5
    batch_size = 32
    epochs_lst = [40, 40, 40, 40, 40]
    votefuns = [weighted_vote,  majority_vote]
    bagging_train_model(n_learners, epochs_lst, batch_size, votefuns, "cnn-bagging.txt")

def test2():
    # n_learners = 2
    votefuns = [weighted_vote,  majority_vote]
    # saved_model_files = ['saved_models/callback-save-30-0.77.hdf5', 'saved_models/callback-save-45-0.77.hdf5',
    #        'saved_models/callback-save-60-0.78.hdf5']
    # keras_cifar10_trained_model_4.h5
    saved_model_files = ['saved_models/keras_cifar10_trained_model_4.h5', 'saved_models/keras_cifar10_trained_model_6.h5']
    n_learners = len(saved_model_files)
    bagging_loading_model(n_learners, saved_model_files, votefuns, "cnn-bagging.txt")

def test3():
    n_learners = 5
    batch_size = 32
    epochs_lst = [40, 40, 40, 40, 40]
    votefuns = [weighted_vote,  majority_vote]
    bagging_train_model(n_learners, epochs_lst, batch_size, votefuns, "cnn-bagging.txt")

def meta_model(n_learners, num_classes):
    # create model
    model = Sequential()
    in_dim = n_learners * num_classes
    print(in_dim)
    model.add(Dense(n_learners*num_classes, input_dim = in_dim, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def snapshot_train_model(epochs, batch_size, M, alpha_zero, name_prefix):
    num_classes = 10
    (x_train, y_train), (x_test, y_test) = load_data()
    y_test_old = y_test[:] # save for error calculation
    (x_train, y_train), (x_test, y_test) = preprocess(x_train, y_train, x_test, y_test)
    model = build_model(x_train, num_classes)
    train_snapshot(x_train, y_train, x_test, y_test, model, batch_size, epochs, M, alpha_zero, name_prefix, data_augmentation=True)

def snapshot_ensemble(epochs, batch_size, M, alpha_zero, name_prefix, meta_epochs):
    snapshot_train_model(epochs, batch_size, M, alpha_zero, name_prefix)
    saved_model_files = []
    for i in range(M):
        saved_model_files.append("snapshot_models/cnn-snapshot-" + str(i+1) + ".h5")
    print(saved_model_files)
    stack_loading_model(saved_model_files, meta_epochs, filename="cnn-snapshot.txt")

if __name__ == "__main__":
    print("Hello UW!")
    # # bagging
    # test1() # bagging for three learners
    # test2() # load saved models
    # test3() # bagging for five learners

    # adaboost for multiple classification
    n_learners = 5
    epochs_lst = [20, 20, 20, 20, 20]
    batch_size = 32
    sample_ratio = 3
    adaboost(n_learners, epochs_lst, batch_size, sample_ratio, "cnn-adaboost.txt")

    '''
    # stack with saved models
    saved_model_files = ['saved_models/keras_cifar10_trained_model_4.h5', 'saved_models/keras_cifar10_trained_model_6.h5']
    meta_epochs = 2
    stack_loading_model(saved_model_files, meta_epochs, filename="cnn-stack.txt")
    '''

    '''
    # stack with trained models
    n_learners = 3;
    epochs_lst = [1, 1, 1];
    batch_size = 32
    meta_epochs = 20
    stack_train_model(n_learners, epochs_lst, batch_size, meta_epochs, filename="cnn-stack.txt")
    '''

    '''
    # snapshot cnn
    epochs = 5
    M = 3
    alpha_zero = 0.0001
    batch_size = 32
    name_prefix = "cnn-snapshot"
    meta_epochs = 20
    snapshot_ensemble(epochs, batch_size, M, alpha_zero, name_prefix, meta_epochs)
    '''
