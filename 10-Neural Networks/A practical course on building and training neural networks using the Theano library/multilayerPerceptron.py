# -*- coding: utf-8 -*-
"""
Created on Sun Jan 03 10:39:34 2016

@author: Gabi
"""

import gzip
import cPickle
import numpy
import theano
import theano.tensor as T

f = gzip.open('C:/nnets/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

train_set = (train_set[0][:5000], train_set[1][:5000])
test_set = (test_set[0][:5000], test_set[1][:5000])

n_train, n_test = map(lambda x:len(x[0]), [train_set, test_set])
dims = train_set[0].shape[1]
n_classes = len(set(train_set[1]))

X = T.dmatrix()
y = T.ivector()

prepare_data = lambda x: (theano.shared(x[0].astype('float64')), theano.shared(x[1].astype('int32')))
training_x, training_y = prepare_data(train_set)
test_x, test_y = prepare_data(train_set)

n_hidden_neurons = 20
W_xh = theano.shared(.01*numpy.random.randn(dims,n_hidden_neurons))
b_xh = theano.shared(numpy.zeros(n_classes))
W_hy = theano.shared(numpy.zeros([n_hidden_neurons,n_classes]))
b_hy = theano.shared(numpy.zeros(n_classes))

h = T.tanh(T.dot(X,W_xh) + b_xh)
y_hat = T.nnet.softmax(T.dot(h,W_hy) + b_hy)
y_pred = T.argmax(y_hat, axis=1)
test_error = T.mean(T.neq(y_pred, y))
training_error = -T.mean(T.log(y_hat)[T.arange(y.shape[0]), y])

learning_rate = 2
updates = [
        [W_xh, W_xh - learning_rate * T.grad(training_error, W_xh)], 
        [b_xh, b_xh - learning_rate * T.grad(training_error, b_xh)],
        [W_hy, W_hy - learning_rate * T.grad(training_error, W_hy)], 
        [b_hy, b_hy - learning_rate * T.grad(training_error, b_hy)],
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

for i in range(100):
    print('Training set mean negative log-likelihood: %f' % training_function())
    print('Test set accuracy: %f' % test_function())
    print('')