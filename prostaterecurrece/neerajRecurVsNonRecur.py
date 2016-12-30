# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 21:49:07 2016

@author: apple
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 05:31:28 2016

@author: apple
"""

import os
os.chdir('/Users/apple/Documents/CodesResearch/CPCTR/cnnRecurVsNonRecur/train_test_101/')

os.environ['THEANO FLAGS'] = 'mode=FAST RUN, device = gpu, floatX=float32'
import theano
import pickle
import timeit
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
#from theano.tensor.nnet.conv import conv2d
#from theano.tensor.signal import pool
from utilitiesCNN import ConvLayer, HiddenLayer, LogisticRegression
import scipy

srng = RandomStreams()
rng = np.random.RandomState(23455)

#loading the dataset
Data = scipy.io.loadmat('train1.mat')
train_set_x = (np.asarray(Data['data'][:1500,:], dtype = theano.config.floatX))
train_set_y = (np.asarray(Data['target'][:1500], dtype = theano.config.floatX))
train_set_y = train_set_y.reshape(1500,)

valid_set_x = (np.asarray(Data['data'][1500:2000,:], dtype = theano.config.floatX))
valid_set_y = (np.asarray(Data['target'][1500:2000], dtype = theano.config.floatX))
valid_set_y = valid_set_y.reshape(500,)

Data = None
del Data

train_set_x = theano.shared(np.asarray(train_set_x, dtype = theano.config.floatX))
train_set_y = theano.shared(np.asarray(train_set_y, dtype = theano.config.floatX))
train_set_y = T.cast(train_set_y, 'int32')


valid_set_x = theano.shared(np.asarray(valid_set_x, dtype = theano.config.floatX))
valid_set_y = theano.shared(np.asarray(valid_set_y, dtype = theano.config.floatX))
valid_set_y = T.cast(valid_set_y, 'int32')

print 'Loading testing data!'
Data= scipy.io.loadmat('test1.mat')

test_set_x = theano.shared(np.asarray(Data['data'][:2000,:],dtype = theano.config.floatX))
test_set_y = (np.asarray(Data['target'][:2000],dtype = theano.config.floatX))
test_set_y = test_set_y.reshape(2000,)
test_set_y= theano.shared(test_set_y)
test_set_y = T.cast(test_set_y,'int32')

''' Building CNN'''
nkerns =[48,48,48]
learning_rate = 0.1
batch_size = 500
n_epochs = 50
x = T.matrix('x')   # the data is presented as rasterized images
y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels
index = T.lscalar()  # index to a [mini]batch

# compute number of minibatches for training, validation and testing
n_train_batches = train_set_x.get_value(borrow=True).shape[0]
n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
n_test_batches = test_set_x.get_value(borrow=True).shape[0]
n_train_batches //= batch_size
n_valid_batches //= batch_size
n_test_batches //= batch_size


conv0_input = x.reshape((batch_size, 3, 101,101))

conv0 = ConvLayer(
        rng,
        input=conv0_input,
        image_shape=(batch_size, 3, 101, 101),
        filter_shape=(nkerns[0], 3, 5, 5),
        poolsize=(2, 2),
        dropout = 0.1,
        parameters = None
    )
    
    
conv1 = ConvLayer(
        rng,
        input=conv0.output,
        image_shape=(batch_size, nkerns[0], 48, 48),
        filter_shape=(nkerns[1], nkerns[0], 6, 6),
        poolsize=(2, 2),
        dropout = 0.2,
        parameters = None
    )
conv2 = ConvLayer(
        rng,
        input=conv1.output,
        image_shape=(batch_size, nkerns[1], 21, 21),
        filter_shape=(nkerns[2], nkerns[1], 7, 7),
        poolsize=(2, 2),
        dropout = 0.2,
        parameters = None
    )

hidden0_input = conv2.output.flatten(2)

hidden0 = HiddenLayer(
        rng,
        input=hidden0_input,
        n_in=nkerns[2] * 7 * 7,
        n_out=500,
        activation=T.tanh,
        dropout=0.5,
        parameters= None
    )
    
#hidden1 = HiddenLayer(
#        rng,
#        input=hidden0.output,
#        n_in=1024,
#        n_out=1024,
#        activation=T.tanh,
#        dropout=0.5,
#        parameters= None
#    )
#    
# Final Logistic regression layer
logisticlayer = LogisticRegression(input=hidden0.output, n_in=500, n_out=2, W= None, b= None)

    # the cost we minimize during training is the NLL of the model
#cost = T.mean(T.sum((logisticlayer.y_pred - y) ** 2), axis =1)

cost = logisticlayer.negative_log_likelihood(y)

params = logisticlayer.params + hidden0.params + conv2.params + conv1.params + conv0.params
# create a list of gradients for all model parameters
grads = T.grad(cost, params)


updates = [(param_i, param_i - learning_rate * grad_i)
            for param_i, grad_i in zip(params, grads)]


train_model = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )
    
# create a function to compute the mistakes that are made by the model
test_model = theano.function(
        [index],
        logisticlayer.errors(y),
        givens={
            x: test_set_x[index * batch_size: (index + 1) * batch_size],
            y: test_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

validate_model = theano.function(
        [index],
        logisticlayer.errors(y),
        givens={
            x: valid_set_x[index * batch_size: (index + 1) * batch_size],
            y: valid_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )


###############
    # TRAIN MODEL #
    ###############
print('... training')
    # early-stopping parameters
patience = 10000  # look as this many examples regardless
patience_increase = 2  # wait this much longer when a new best is
                           # found
improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

best_validation_loss = np.inf
best_iter = 0
test_score = 0.
start_time = timeit.default_timer()

epoch = 0
done_looping = False

while (epoch < n_epochs) and (not done_looping):
    train_set_x = None
    train_set_y = None
    del train_set_x,train_set_y
    
    valid_set_x = None
    valid_set_y = None
    del valid_set_x,valid_set_y
    
    test_set_x = None
    test_set_y = None
    del test_set_x,test_set_y
    
    if (epoch % 22) == 0 or epoch == 0:
        Dataindex = 1
    else:
        Dataindex = Dataindex + 1
        
    if (epoch % 5) == 0 or epoch == 0:
        testindex = 1
    else:
        testindex = testindex + 1
    
    filename = 'train' + str(Dataindex) + '.mat'
    validname = 'train' + str(Dataindex) + '.mat'
    testname = 'test' + str(testindex) + '.mat'
    
    Data = scipy.io.loadmat(filename)

    train_set_x = Data['data']
    train_set_y = Data['target']
    Data = None
    del Data
    
    train_set_x = train_set_x.astype('float32')
    train_set_y = train_set_y.reshape(train_set_y.shape[0],)
    train_set_y = train_set_y.astype('int32')

    train_set_x = theano.shared(np.asarray(train_set_x, dtype = theano.config.floatX))
    train_set_y = theano.shared(np.asarray(train_set_y, dtype = theano.config.floatX))
    train_set_y = T.cast(train_set_y, 'int32')

    
    Data = scipy.io.loadmat(validname)

    valid_set_x = Data['data']
    valid_set_y = Data['target']


    valid_set_x = valid_set_x.astype('float32')
    valid_set_y = valid_set_y.reshape(valid_set_y.shape[0],)
    valid_set_y = valid_set_y.astype('int32')

    valid_set_x = theano.shared(np.asarray(valid_set_x, dtype = theano.config.floatX))
    valid_set_y = theano.shared(np.asarray(valid_set_y, dtype = theano.config.floatX))
    valid_set_y = T.cast(valid_set_y, 'int32')
    
    
    Data = scipy.io.loadmat(testname)

    test_set_x = Data['data']
    test_set_y = Data['target']
#    X_test = X_test[:30000,:]
#    Y_test = Y_test[:30000]

    test_set_x = test_set_x.astype('float32')
    test_set_y = test_set_y.reshape(test_set_y.shape[0],)
    test_set_y = test_set_y.astype('int32')

    test_set_x = theano.shared(np.asarray(test_set_x, dtype = theano.config.floatX))
    test_set_y = theano.shared(np.asarray(test_set_y, dtype = theano.config.floatX))
    test_set_y = T.cast(test_set_y, 'int32')
    
    print 'Epoch #'+str(epoch)+' training data '+ str(Dataindex) +' validation data '+str(Dataindex)+' testing data '+ str(testindex)    

    
    
    epoch = epoch + 1
    for minibatch_index in range(n_train_batches):

        iter = (epoch - 1) * n_train_batches + minibatch_index

        if iter % 100 == 0:
            print('training @ iter = ', iter)
        cost_ij = train_model(minibatch_index)

        if (iter + 1) % validation_frequency == 0:

                # compute zero-one loss on validation set
            validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
            this_validation_loss = np.mean(validation_losses)
            print('epoch %i, minibatch %i/%i, validation error %f %%' %
                      (epoch, minibatch_index + 1, n_train_batches,
                       this_validation_loss * 100.))

                # if we got the best validation score until now
            if this_validation_loss < best_validation_loss:
                pickle.dump(params, open( "params_rec_pred_101.p", "wb" ) )
                pickle.dump(dict(Y=y,features=conv1.output.flatten(2)),open('features_rec_pred_101.p','wb'))
                    #improve patience if loss improvement is good enough
                if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
                    patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [
                        test_model(i)
                        for i in range(n_test_batches)
                    ]
                    test_score = np.mean(test_losses)
                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

        if patience <= iter:
            done_looping = True
            break

end_time = timeit.default_timer()
print('Optimization complete.')
print('Best validation score of %f %% obtained at iteration %i, '
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
#print(('The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)



