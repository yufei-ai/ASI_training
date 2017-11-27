# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 10:39:34 2016

@author: Gabi
"""

import gzip
import cPickle

f = gzip.open('C:/nnets/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

n_train, n_test = map(lambda x:len(x[0]), [train_set, test_set])
dims = train_set[0].shape[1]
n_classes = len(set(train_set[1]))


import numpy
import theano
import theano.tensor as T

X = T.dmatrix()
y = T.ivector()

prepare_data = lambda x: (theano.shared(x[0].astype('float64')), theano.shared(x[1].astype('int32')))
(training_x, training_y), (test_x, test_y) = map(prepare_data, [train_set, test_set])

W = theano.shared(numpy.zeros([dims,n_classes]))
b = theano.shared(numpy.zeros(n_classes))

y_hat = T.nnet.softmax(T.dot(X,W) + b)
y_pred = T.argmax(y_hat, axis=1)
test_error = T.mean(T.neq(y_pred, y))
training_error = -T.mean(T.log(y_hat)[T.arange(y.shape[0]), y])

learning_rate = .5
updates = [
        [W, W - learning_rate * T.grad(training_error, W)], 
        [b, b - learning_rate * T.grad(training_error, b)]
    ]

idx = T.ivector()
training_function = theano.function(
    inputs = [idx],
    outputs = training_error,
    updates = updates,
    givens = {X:training_x[idx], y: training_y[idx]}
    )

test_function = theano.function(
    inputs = [],
    outputs = test_error,
    givens = {X: test_x, y: test_y}
    )
    
getMiniBatches = (lambda n, colLen: numpy.reshape(numpy.random.permutation(n)[:n//colLen*colLen], [n//colLen, colLen]))

minibatchSize = 50000
for dataset_cycles in range(50):
    for minibatch_idx in getMiniBatches(n_train, minibatchSize):
        training_function(minibatch_idx)
    print('Test set accuracy: %f' % test_function())