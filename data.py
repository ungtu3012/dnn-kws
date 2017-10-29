import utils
import random
import numpy as np

X_train, Y_train, X_test, Y_test = None, None, None, None

def load():
    global X_train
    global Y_train
    global X_test
    global Y_test

    X_train = np.load('train_x.npy')
    Y_train = np.load('train_y.npy')
    X_test = np.load('test_x.npy')
    Y_test = np.load('test_y.npy')

# def train_batch_generator(batchsize=128, show_original=False):
#     X_batch = []
#     Y_batch = []

#     while 1:
#         for i in range(len(X_train)):
#             X_batch.append(X_train[i])
#             Y_batch.append(Y_train[i])
#             if show_original:
#                 print('original: ', utils.unvectorize_y(Y_train[i]))
            
#             if len(X_batch) >= batchsize:
#                 X_batch, seq_len_batch = utils.pad_sequences(X_batch)
#                 Y_batch = utils.sparse_tuple_from(Y_batch)
#                 yield X_batch, seq_len_batch, Y_batch
#                 X_batch = []
#                 Y_batch = []

def train_batch_generator(batchsize=128, show_original=False):
    X_batch = []
    Y_batch = []

    while 1:
        for i in range(len(X_train)):
            X_batch.append(X_train[i])
            Y_batch.append(Y_train[i])
            if show_original:
                print('original: ', utils.unvectorize_y(Y_train[i]))
            
            if len(X_batch) >= batchsize:
                yield np.array(X_batch), np.array(Y_batch)
                X_batch = []
                Y_batch = []

# def test_batch_generator(batchsize=128, show_original=False):
#     X_batch = []
#     Y_batch = []

#     while 1:
#         for i in range(len(X_test)):
#             X_batch.append(X_test[i])
#             Y_batch.append(Y_test[i])
#             if show_original:
#                 print('original: ', utils.unvectorize_y(Y_train[i]))
            
#             if len(X_batch) >= batchsize:
#                 X_batch, seq_len_batch = utils.pad_sequences(X_batch)
#                 Y_batch = utils.sparse_tuple_from(Y_batch)
#                 yield X_batch, seq_len_batch, Y_batch
#                 X_batch = []
#                 Y_batch = []

def test_batch_generator(batchsize=128, show_original=False):
    X_batch = []
    Y_batch = []

    while 1:
        for i in range(len(X_test)):
            X_batch.append(X_test[i])
            Y_batch.append(Y_test[i])
            if show_original:
                print('original: ', utils.unvectorize_y(Y_train[i]))
            
            if len(X_batch) >= batchsize:
                yield X_batch, Y_batch
                X_batch = []
                Y_batch = []

def stats():
    return len(X_train), len(X_test)

# load()