import tensorflow as tf
from tensorflow.keras.datasets import cifar10
import numpy as np

import argparse
import importlib
import csv
import pdb

parser = argparse.ArgumentParser(
        description = 'Specify key arguments for this project.')
parser.add_argument(
        '--model', default = 'resnet20', 
        help = 'Name of loaded model.')
parser.add_argument(
        '--class_num', default = 10, type = int,
        help = 'Number of output class.')
parser.add_argument(
        '--dataset', default = 'cifar10', 
        help = 'Dataset.')
parser.add_argument(
        '--quantilize', default = 'False',
        help = 'If quantilize model.')
parser.add_argument(
        '--quantilize_w', default = 32, type = int,
        help = 'Weights bits width for quantilize model ')
parser.add_argument(
        '--quantilize_x', default = 32, type = int,
        help = 'Activation bits width for quantilize model ')
parser.add_argument(
        '--weight_decay', default = 0.0005, type = float,
        help = 'Weight decay for regularizer l2.')
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
        '--ckpt_dir', default = 'resnet20', 
        help = 'Directory of checkpoint.')
parser.add_argument(
        '--log_dir', default = 'log_dir',
        help = 'Directory of log file. Always in `log` directory.')
parser.add_argument(
        '--log_file', default = 'log_file.txt',
        help = 'Name of log file.')
args = parser.parse_args()

# Param
lr_drop = 20
lr_decay = 1e-6
learning_rate = args.learning_rate
batch_size = args.batch_size
num_epochs = args.num_epochs

def normalize(X_train,X_test):
    mean    = np.mean(X_train,axis=(0, 1, 2, 3))
    std     = np.std(X_train, axis=(0, 1, 2, 3))
    X_train = (X_train-mean)/(std+1e-7)
    X_test  = (X_test-mean)/(std+1e-7)
    return X_train, X_test


def lr_scheduler(epoch):
    if epoch < 7800:
        return 0.1
    elif epoch < 31200:
        return 0.01
    elif epoch < 54600:
        return 0.001
    elif epoch < 78000:
        return 0.0001
    else:
        return 0.00001
    # return learning_rate * (0.5 ** (epoch // lr_drop))


if __name__ == '__main__':
    if args.dataset == 'cifar10':
        # The data, shuffled and split between train and test sets:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()
        x_train = x_train.astype('float32')
        x_test  = x_test.astype('float32')

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

    # model.compile(
            # loss='categorical_crossentropy', optimizer=sgd,
            # metrics=['accuracy'], run_eagerly = True)
    model.compile(
            loss='categorical_crossentropy', optimizer=sgd,
            metrics=['accuracy'])

    # training process in a for loop with learning rate drop every 20 epoches.
    historytemp = model.fit(
            datagen.flow(x_train, y_train, batch_size = batch_size), 
            steps_per_epoch=x_train.shape[0] // batch_size, 
            epochs=num_epochs, 
            validation_data=(x_test, y_test),
            callbacks=[reduce_lr],
            verbose=2)

    # log_path = './log/' + args.log_dir + '/' + args.log_file
    # with open(log_path, 'w', newline = '') as csvfile:
        # csvwriter = csv.DictWriter(csvfile, historytemp.history.keys())
        # csvwriter.writeheader()
        # csvwriter.writerow(historytemp.history)
