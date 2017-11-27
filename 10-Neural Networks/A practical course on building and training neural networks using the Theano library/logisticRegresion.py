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

training_function = theano.function(
    inputs = [],
    outputs = training_error,
    updates = updates,
    givens = {X:training_x, y: training_y}
    )

test_function = theano.function(
    inputs = [],
    outputs = test_error,
    givens = {X: test_x, y: test_y}
    )

for i in range(300):
    print('Training set negative log-likelihood: %f' % training_function())
    print('Test set accuracy: %f' % test_function())
    print('')
    
classify = theano.function(
    inputs = [],
    outputs = y_pred,
    givens = {X: test_x}
    )    
    
from plot_mnist_image import plot_mnist_digit
test_labels = classify()
for i in range(10):
    plot_mnist_digit(test_x.get_value()[i])
    print test_labels[i], test_y.get_value()[i]