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
(training_x, training_y), (test_x, test_y), (validation_x, validation_y) = map(prepare_data, [train_set, test_set, valid_set])

W = theano.shared(numpy.zeros([dims,n_classes]))
b = theano.shared(numpy.zeros(n_classes))

y_hat = T.nnet.softmax(T.dot(X,W) + b)
y_pred = T.argmax(y_hat, axis=1)
test_error = T.mean(T.neq(y_pred, y))
training_error = -T.mean(T.log(y_hat)[T.arange(y.shape[0]), y])

learning_rate = 0.5
params = [W, b]
updates = []
ada_rates = {}
for p in params:
    ada_rates[p] = theano.shared(1.+0.*p.get_value())
    updates += [(p, p - learning_rate * ada_rates[p] * T.grad(training_error, p))]
    

idx = T.ivector()
training_function = theano.function(
    inputs = [idx],
    outputs = [training_error]+[T.grad(training_error, p) for p in params],
    updates = updates,
    givens = {X:training_x[idx], y: training_y[idx]}
    )

test_function = theano.function(
    inputs = [],
    outputs = test_error,
    givens = {X: test_x, y: test_y}
    )
    
validation_function = theano.function(
    inputs = [],
    outputs = test_error,
    givens = {X: validation_x, y: validation_y}
    )    
    
getMiniBatches = (lambda n, colLen: numpy.reshape(numpy.random.permutation(n)[:n//colLen*colLen], [n//colLen, colLen]))

minibatchSize = 600

nEpoch = 0
minEpochs = 50
patience = 0
maxPatience = 20

bestScore = numpy.inf
lastGradients = None; curGradients = None

while nEpoch < minEpochs or patience < maxPatience:
    for minibatch_idx in getMiniBatches(n_train, minibatchSize):
        lastGradients = curGradients
        trFunOut = training_function(minibatch_idx)
        curGradients = dict(zip(params,trFunOut[1:]))
        if lastGradients is not None:
            for p in params:
                g = lastGradients[p] * curGradients[p]
                ar = numpy.copy(ada_rates[p].get_value())
                ar[g>0] += .05; ar[g<0] *= .95
                ada_rates[p].set_value(ar)
        
    validScore = validation_function()
    if validScore < bestScore:
        bestScore = validScore
        patience = 0
        bestParams = {i: i.get_value() for i in params}
    else: patience += 1
    print('Epoch: %d, Validation set accuracy: %f, Test set accuracy: %f, Patience: %d' % (nEpoch, validScore, test_function(), patience))
    nEpoch += 1

for var, val in bestParams.iteritems():
    var.set_value(val)
    