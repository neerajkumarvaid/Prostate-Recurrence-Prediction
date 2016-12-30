# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 05:36:12 2016

@author: apple
"""
import os
#('/Users/neerajkumar/Documents/DeepLearningTutorials/TheanoTut/')
os.environ['THEANO FLAGS'] = 'mode=FAST RUN, device = gpu, floatX=float32'
#import sys
#import timeit
#import pickle
import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
#import cPickle
#import gzip
#import random
#import h5py
#import scipy.io

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

srng = RandomStreams()
class ConvLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, parameters,image_shape, dropout,poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height, filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows, #cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        if parameters is None:
            fan_in = numpy.prod(filter_shape[1:])
#        # each unit in the lower layer receives a gradient from:
#        # "num output feature maps * filter height * filter width" /
#        #   pooling size
            fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
#        # initialize weights with random weights
            W_bound = numpy.sqrt(6. / (fan_in + fan_out))
            self.W = theano.shared(
                   numpy.asarray(
                   rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
                    dtype=theano.config.floatX
                    ),
                    borrow=True
                    )
        
        # the bias is a 1D tensor -- one bias per output feature map
            b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
            self.b = theano.shared(value=b_values, borrow=True)
        else:
            self. W = parameters[0]
            self.b = parameters[1]
        # convolve input feature maps with filters
        conv_out = conv.conv2d(
            input=input,
            filters=self.W,
            filter_shape=filter_shape,
            image_shape=image_shape
        )

        # downsample each feature map individually, using maxpooling
        pooled_out = pool.pool_2d(
            input=conv_out,
            ds=poolsize,
            ignore_border=True
        )
        
        #Applying Dropout              
        if dropout > 0.0:
            retain_prob = 1 - dropout
            pooled_out *= srng.binomial(pooled_out.shape, p=retain_prob, dtype=theano.config.floatX)
            pooled_out /= retain_prob

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1, n_filters, 1, 1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height

# This is for tanh activation
        #self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

# This is for Relu activation
        self.output = pooled_out + self.b.dimshuffle('x', 0, 'x', 'x')
        self.output = theano.tensor.switch(self.output<0, 0, self.output)

        # store parameters of this layer
        self.params = [self.W, self.b]

        # keep track of model input
        self.input = input
        
        
class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out,dropout,parameters,activation
                 ):
      
        self.input = input

        if parameters is None:
            W_values = numpy.asarray(rng.uniform(
                    low=-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            #if activation == theano.tensor.nnet.sigmoid:
             #   W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
            self.W = W
            self.b = b
        else:
           self.W = parameters[0]
           self.b = parameters[1]

     
            

        lin_output = T.dot(input, self.W) + self.b.dimshuffle('x',0)
        #out = theano.tensor.switch(lin_output<0, 0, lin_output)
        #out = T.tanh(lin_output)
       # out = (theano.tensor.switch(lin_output<0, 0, lin_output) if activation is 'relu')
        #              else activation(lin_output))
        if activation is 'relu': 
            out = (theano.tensor.switch(lin_output<0, 0, lin_output))
        else:
            out = T.tanh(lin_output)                
                       
        # Applying dropout
        if dropout > 0.0:
            srng = theano.tensor.shared_randomstreams.RandomStreams(
                rng.randint(999999))
    # p=1-p because 1's indicate keep and p is prob of dropping
            mask = srng.binomial(n=1, p=1-dropout, size=out.shape)
    # The cast is important because
    # int * float32 = float64 which pulls things off the gpu
            self.output = out * T.cast(mask, theano.config.floatX)
        else:
            self.output = out
        # parameters of the model
        self.params = [self.W, self.b]
        
        
class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W, b):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
                W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                         dtype=theano.config.floatX),
                                        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        if b is None:
                b = theano.shared(value=numpy.zeros((n_out,),
                                                         dtype=theano.config.floatX),
                                       name='b', borrow=True)
       
        self.W = W
        self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
                \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        #z = -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
        #zk = theano.tensor.scalar('zk')
        #zp = theano.printing.Print('this is a very important value')(zk)
        #f = theano.function([zk],zp)
        #z = theano.shared(z)
        #f(z)
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', target.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()
        
class LinearRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W, b):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
                W = theano.shared(value=numpy.zeros((n_in, n_out),
                                                         dtype=theano.config.floatX),
                                        name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        if b is None:
                b = theano.shared(value=numpy.zeros((n_out,),
                                                         dtype=theano.config.floatX),
                                       name='b', borrow=True)
       
        self.W = W
        self.b = b

        # compute vector of class-membership probabilities in symbolic form
        self.y_pred = T.dot(input, self.W) + self.b

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred = T.switch(self.y_pred<0,0,self.y_pred)

        # parameters of the model
        self.params = [self.W, self.b]

    def mean_squared_error(self,y):
        return T.mean(T.sum(T.sqr(self.y_pred - y),axis = 1))
        

        
        

