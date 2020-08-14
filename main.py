import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

import argparse
import importlib
import pdb

parser = argparse.ArgumentParser(
        description = 'Specify key arguments for this project.')
parser.add_argument(
        '--batch_size', default = 128, type = int,
        help = 'Numbers of images to process in a batch.')
parser.add_argument(
        '--num_epochs', default = 250, type = int, 
        help = 'Number of epochs to train. -1 for unlimited.')
parser.add_argument(
        '--learning_rate', default = 0.1, type = float,
        help = 'Initial learning rate used.')
parser.add_argument(
        '--model', default = 'resnet20', 
        help = 'Name of loaded model.')
parser.add_argument(
        '--ckpt_dir', default = 'resnet20', 
        help = 'Directory of checkpoint.')
parser.add_argument(
        '--dataset', default = 'cifar10', 
        help = 'Dataset.')
args = parser.parse_args()

# Param
lr_drop = 20
lr_decay = 1e-6
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs

def normalize(X_train,X_test):
    # this function normalize inputs for zero mean and unit variance
    # it is used when training a model.
    # Input: training set and test set
    # Output: normalized training set and test set according to the trianing set statistics.
    mean = np.mean(X_train,axis=(0, 1, 2, 3))
    std = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test = (X_test-mean)/(std+1e-7)
    return X_train, X_test


def lr_scheduler(epoch):
    return learning_rate * (0.5 ** (epoch // lr_drop))


if __name__ == '__main__':
    if args.dataset == 'cifar10':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')

        # Normalization 
        x_train, x_test = normalize(x_train, x_test)

        y_train = tf.keras.utils.to_categorical(y_train, 10)
        y_test  = tf.keras.utils.to_categorical(y_test, 10)


    # model = cifar10vgg(False)
    # y_pred = model.predict(x_test)
    # m = tf.keras.metrics.Accuracy()
    # m.update_state(np.argmax(y_pred,1), np.argmax(y_test,1))
    # pred_accuracy = m.result().numpy()
    # pdb.set_trace()

    reduce_lr = tf.keras.callbacks.LearningRateScheduler(lr_scheduler)

    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images
    datagen.fit(x_train)

    #optimization details
    sgd = tf.keras.optimizers.SGD(
            lr=learning_rate, decay=lr_decay, momentum=0.9, nesterov=True)
    model = importlib.import_module(
            '.' + args.model, 'models').model 
    model.compile(
            loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 25 epoches.
    # historytemp = model.fit_generator(
    historytemp = model.fit(
            datagen.flow(x_train, y_train, batch_size = batch_size), 
            steps_per_epoch=x_train.shape[0] // batch_size, 
            epochs=num_epochs, 
            validation_data=(x_test, y_test),
            callbacks=[reduce_lr],
            verbose=2)
    # pdb.set_trace()

    # predicted_x = model.predict(x_test)
    # residuals = np.argmax(predicted_x,1)!=np.argmax(y_test,1)

    # loss = sum(residuals)/len(residuals)
    # print("the validation 0/1 loss is: ",loss)
