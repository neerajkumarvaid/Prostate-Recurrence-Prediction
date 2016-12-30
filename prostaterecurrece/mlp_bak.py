
"""
Classes for various types of hidden layers

Created on Sun Jul 5 13:04:13 2015

@author: neerajkumar
"""

import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
#srng = RandomStreams()
def ReLU(x):
    y = T.maximum(0.0, x)
    return(y)
#### sigmoid
def Sigmoid(x):
    y = T.nnet.sigmoid(x)
    return(y)

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


class BinarizationLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
    
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low=0,#-numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        
        self.binary_output = T.ge(self.output, 0.5*T.ones_like(self.output))
        # parameters of the model
        self.params = [self.W, self.b]

    def xent(self, y):
        
        return ((-(self.C*y*T.log(self.output) + (self.C/(self.C-1))*(1-y)*T.log(1-self.output)).mean(axis=0))).mean()
        #return ((-(y*T.log(self.output) + T.log(1-self.output)).mean(axis=0))).mean()


    def Wtxent(self, y, w, k):
        
        c1 = T.cast(T.sum(y),'float32')  
        c2 = T.cast(T.sum(1-y),'float32')  
        self.C =  (c1+c2)/c1
        
        return ((-((2-T.nnet.sigmoid(k))*self.C*y*w*T.log(self.output) + (2+T.nnet.sigmoid(k))*(self.C/(self.C-1))*(1-y)*T.log(1-self.output)).mean(axis=0))).mean()
        
        
       
    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.binary_output, y))
        else:
            raise NotImplementedError()
            
    def sensitivity(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            a = T.cast(T.sum(y*self.binary_output), 'float32') 
            return a/(T.sum(y)  )
        else:
            raise NotImplementedError()
            
    def specificity(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            a = T.cast(T.sum((1-y)*(1-self.binary_output)),'float32') 
            return a/(T.sum(1-y)  )
        else:
            raise NotImplementedError()

class SigmoidLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.nnet.sigmoid):
    
        self.input = input

        if W is None:
            W_values = numpy.asarray(rng.uniform(
                    low= -numpy.sqrt(6. / (n_in + n_out)),
                    high=numpy.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]

    def xent(self, y,v):
        
        return ((-(y*T.log(self.output) + (1-y)*T.log(1-self.output)).mean(axis=1))*v).mean()



class L2SVMLayer(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
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
        self.p_y_given_x = T.dot(input, self.W) + self.b

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.output= T.argmax(self.p_y_given_x, axis=1)

        # parameters of the model
        self.params = [self.W, self.b]

    def L2SVMcost(self, y):
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

        '''p = -T.ones_like((y.shape[0],7))
        
        result, updates = theano.scan(fn = lambda p,y: T.basic.set_subtensor(p[i,y[i]]=1),
                                                                outputs_info = -T.ones_like((y.shape[0],7)),
                                                                non_sequences = y,
                                                                n_steps = y.shape[0])
        final_result = result[-1]
        f = theano.function([y,p],final_result,updates = updates)
                                                                
        for i in xrange(500):
                p = T.basic.set_subtensor(p[i,y[i]]=1)
        print p.shape
        print f(y,p)
        print f(y,p).shape'''
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
        z = 0.5*T.dot( T.flatten(self.W,outdim=1), T.flatten(self.W, outdim=1)) + 0.5*T.dot( T.flatten(self.b,outdim=1), T.flatten(self.b, outdim=1)) +0.6* T.sum(T.maximum(0,(1-self.p_y_given_x *y)),axis=1).mean()
        #zk = theano.tensor.scalar('zk')
        #zp = theano.printing.Print('this is a very important value')(zk)
        #f = theano.function([zk],zp)
        #z = theano.shared(z)
        #f(z)
        return z

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # check if y has same dimension of y_pred
        if y.ndim != self.output.ndim:
            raise TypeError('y should have the same shape as self.y_pred',
                ('y', y.type, 'y_pred', self.output.type))
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.output, y))
        else:
            raise NotImplementedError()


