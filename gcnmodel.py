from __future__ import print_function
import pdb
import numpy as np
import sys
from os import path
import scipy as sp
import theano
from lasagne.utils import floatX
import theano.tensor as T
import lasagne
from lasagne.regularization import regularize_layer_params_weighted, l2, l1
from lasagne.regularization import regularize_layer_params
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
from sklearn.preprocessing import normalize
import logging
import json
import codecs
import pickle
import gzip
from collections import OrderedDict
from _collections import defaultdict
from binstar_client.tests.coverage_report import report
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)


'''
These sparse classes are copied from https://github.com/Lasagne/Lasagne/pull/596/commits
'''
class SparseInputDenseLayer(DenseLayer):
    '''
    An input layer for sparse input and dense output data.
    '''
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        #activation = S.dot(input, self.W)
        activation = S.structured_dot(input, self.W)
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class SparseInputDropoutLayer(DropoutLayer):
    '''
    A dropout layer for sparse input data, note that this layer
    can not be applied to the output of SparseInputDenseLayer
    because the output of SparseInputDenseLayer is dense.
    '''
    def get_output_for(self, input, deterministic=False, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        if deterministic or self.p == 0:
            return input
        else:
            # Using Theano constant to prevent upcasting
            one = T.constant(1, name='one')
            retain_prob = one - self.p

            if self.rescale:
                input = S.mul(input, one/retain_prob)

            input_shape = self.input_shape
            if any(s is None for s in input_shape):
                input_shape = input.shape

            return input * self._srng.binomial(input_shape, p=retain_prob,
                                               dtype=input.dtype)
        
class SparseConvolutionDenseLayer(DenseLayer):
    '''
    A graph convolutional layer where input is sparse and output is dense
    '''
    def __init__(self, incoming, A=None, **kwargs):
        super(SparseConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.A = A

        
    def get_output_for(self, input, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")
        
        activation = S.structured_dot(input, self.W)
        #do the convolution
        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class ConvolutionDenseLayer(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    '''

    def __init__(self, incoming, A=None, **kwargs):
        super(ConvolutionDenseLayer, self).__init__(incoming, **kwargs)
        self.A = A
    
    def get_output_for(self, input, **kwargs):
        target_indices = kwargs.get('target_indices') 
        activation = T.dot(input, self.W)
        #do the convolution
        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class ConvolutionDenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(ConvolutionDenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices
    
    def get_output_for(self, input, A=None, target_indices=None, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution

        if A:
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class ConvolutionDenseLayer3(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, **kwargs):
        super(ConvolutionDenseLayer3, self).__init__(incoming, **kwargs)
    
    def get_output_for(self, input, A=None, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution

        if A:
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        return self.nonlinearity(activation)

class ConvolutionDenseLayer_zero(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, A=None, **kwargs):
        super(ConvolutionDenseLayer_zero, self).__init__(incoming, **kwargs)
        self.A = A
    
    def get_output_for(self, input, **kwargs):
        activation = T.dot(input, self.W)
        #do the convolution
        
        activation = S.structured_dot(self.A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)

class ConvolutionLayer(lasagne.layers.Layer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, A=None, nonlinearity=lasagne.nonlinearities.linear, **kwargs):
        super(ConvolutionLayer, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices
        self.A = A
        self.nonlinearity = nonlinearity
    
    def get_output_for(self, input, target_indices=None, **kwargs):
        #do the convolution
        activation = S.structured_dot(self.A, input)


        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)

class DenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input and output are both dense.
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''

    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(DenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices
    
    def get_output_for(self, input, target_indices=None, **kwargs):
        activation = T.dot(input, self.W)
        
        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)
        if  self.use_target_indices and target_indices:
            activation = activation[target_indices, :]
        return self.nonlinearity(activation)


class SparseConvolutionDenseLayer2(DenseLayer):
    '''
    A graph convolutional layer where input is sparse and output is dense
    In this class H is passed as argument to get_output instead of being
    the parameter of the layer.
    '''
    def __init__(self, incoming, use_target_indices=False, **kwargs):
        super(SparseConvolutionDenseLayer2, self).__init__(incoming, **kwargs)
        self.use_target_indices = use_target_indices

        
    def get_output_for(self, input, A=None, **kwargs):
        if not isinstance(input, (S.SparseVariable, S.SparseConstant,
                                  S.sharedvar.SparseTensorSharedVariable)):
            raise ValueError("Input for this layer must be sparse")

        
        activation = S.structured_dot(input, self.W)
        if A:
            #do the convolution
            activation = S.structured_dot(A, activation)

        if self.b is not None:
            activation = activation + self.b.dimshuffle('x', 0)

        return self.nonlinearity(activation)


class MultiplicativeGatingLayer(lasagne.layers.MergeLayer):
    """
    Generic layer that combines its 3 inputs t, h1, h2 as follows:
    y = t * h1 + (1 - t) * h2
    """
    def __init__(self, gate, input1, input2, **kwargs):
        incomings = [gate, input1, input2]
        super(MultiplicativeGatingLayer, self).__init__(incomings, **kwargs)
        assert gate.output_shape == input1.output_shape == input2.output_shape
    
    def get_output_shape_for(self, input_shapes):
        return input_shapes[0]
    
    def get_output_for(self, inputs, **kwargs):
        return inputs[0] * inputs[1] + (1.0 - inputs[0]) * inputs[2]

def highway_dense(incoming, gconv=False, 
                  #Wh=lasagne.init.Orthogonal(),
                  Wh=lasagne.init.GlorotUniform(), 
                  bh=lasagne.init.Constant(0.0),
                  #Wt=lasagne.init.Orthogonal(),
                  Wt=lasagne.init.GlorotUniform(), 
                  bt=lasagne.init.Constant(-4.0),
                  nonlinearity=lasagne.nonlinearities.sigmoid, **kwargs):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    #bt should be set to -2 according to http://people.idsia.ch/~rupesh/very_deep_learning/ and kim et al 2015
    # regular layer
    #l_h = nn.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    if gconv:
        l_h = ConvolutionDenseLayer2(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    else:
        l_h = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wh, b=bh, nonlinearity=nonlinearity)
    # gate layer
    l_t = lasagne.layers.DenseLayer(incoming, num_units=num_inputs, W=Wt, b=bt,
                                   nonlinearity=T.nnet.sigmoid)
        
    return MultiplicativeGatingLayer(gate=l_t, input1=l_h, input2=incoming), l_t

def residual_dense(incoming, nonlinearity=lasagne.nonlinearities.selu):
    num_inputs = int(np.prod(incoming.output_shape[1:]))
    convX = ConvolutionDenseLayer2(incoming, num_units=num_inputs, nonlinearity=None)
    convX_plus_X = lasagne.layers.ElemwiseSumLayer([convX, incoming], coeffs=1, cropping=None)
    return lasagne.layers.NonlinearityLayer(convX_plus_X, nonlinearity=nonlinearity)
                                                    
    

def np_softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]


class GraphConv():
    '''
    A general theano-based graph convolutional neural network model based in Kipf (2016).
    Note that the input is assumed to be sparse (as in BoW model of text).
    '''
    def __init__(self, input_size, output_size, hid_size_list, regul_coef, drop_out, dtype='float32', batchnorm=False, highway=True):
        
        self.input_size = input_size
        self.output_size = output_size
        self.hid_size_list = hid_size_list
        self.regul_coef = regul_coef
        self.drop_out = drop_out
        self.dtype=dtype
        self.dtypeint = 'int64' if self.dtype == 'float64' else 'int32'
        self.fitted = False
        self.batchnorm = batchnorm
        self.highway = highway
        logging.info('highway is {}'.format(self.highway))
    
    def build_model(self, A, use_text=True, use_labels=True, seed=77):
        np.random.seed(seed)
        logging.info('Graphconv model input size {}, output size {} and hidden layers {} regul {} dropout {}.'.format(self.input_size, self.output_size, str(self.hid_size_list), self.regul_coef, self.drop_out))
        self.X_sym = S.csr_matrix(name='inputs', dtype=self.dtype)
        self.train_indices_sym = T.lvector()
        self.dev_indices_sym = T.lvector()
        self.test_indices_sym = T.lvector()
        self.A_sym = S.csr_matrix(name='NormalizedAdj', dtype=self.dtype)
        self.train_y_sym = T.lvector()
        self.dev_y_sym = T.lvector()
        #nonlinearity = lasagne.nonlinearities.rectify 
        #Wh = lasagne.init.GlorotUniform(gain='relu')
        nonlinearity = lasagne.nonlinearities.tanh 
        Wh = lasagne.init.GlorotUniform(gain=1)

        #input layer
        l_in = lasagne.layers.InputLayer(shape=(None, self.input_size),
                                         input_var=self.X_sym)
        l_hid = SparseInputDenseLayer(l_in, num_units=self.hid_size_list[0], nonlinearity=nonlinearity)

        #add hidden layers

        l_hid = lasagne.layers.dropout(l_hid, p=self.drop_out)
        num_inputs_txt = int(np.prod(l_hid.output_shape[1:]))         
        Wt_txt = lasagne.init.Orthogonal()
        self.gate_layers = []
        logging.info('{} gconv layers'.format(len(self.hid_size_list)))
        if len(self.hid_size_list) > 1:
            for i, hid_size in enumerate(self.hid_size_list):
                if i == 0:
                    #we have already added the first hidden layer which is nonconvolutional
                     continue
                else:
                    if self.highway:
                        l_hid, l_t_hid = highway_dense(l_hid, gconv=True, nonlinearity=nonlinearity, Wt=Wt_txt, Wh=Wh)
                        self.gate_layers.append(l_t_hid)
                    else:
                        l_hid = ConvolutionDenseLayer2(l_hid, num_units=hid_size, nonlinearity=nonlinearity)

        self.l_out = ConvolutionDenseLayer3(l_hid, num_units=self.output_size, nonlinearity=lasagne.nonlinearities.softmax)
        self.output = lasagne.layers.get_output(self.l_out, {l_in:self.X_sym}, A=self.A_sym, deterministic=False)
        self.train_output = self.output[self.train_indices_sym, :]
        self.train_pred = self.train_output.argmax(-1)
        self.dev_output = self.output[self.dev_indices_sym, :]
        self.dev_pred = self.dev_output.argmax(-1)
        self.train_acc = T.mean(T.eq(self.train_pred, self.train_y_sym))
        self.dev_acc = T.mean(T.eq(self.dev_pred, self.dev_y_sym))
        self.train_loss = lasagne.objectives.categorical_crossentropy(self.train_output, self.train_y_sym).mean()
        if self.regul_coef > 0:
            #add l1 regularization
            self.train_loss += lasagne.regularization.regularize_network_params(self.l_out, penalty=lasagne.regularization.l1) * self.regul_coef
            #add l2 regularization
            self.train_loss += lasagne.regularization.regularize_network_params(self.l_out, penalty=lasagne.regularization.l2) * self.regul_coef

        self.dev_loss = lasagne.objectives.categorical_crossentropy(self.dev_output, self.dev_y_sym).mean()
        
        #deterministic output
        self.determ_output = lasagne.layers.get_output(self.l_out, {l_in:self.X_sym}, A=self.A_sym, deterministic=True)
        self.test_output = self.determ_output[self.test_indices_sym, :]
        self.test_pred = self.test_output.argmax(-1)

        self.gate_outputs = []
        self.f_gates = []
        for i, l in enumerate(self.gate_layers):

            self.gate_outputs.append(lasagne.layers.get_output(l, {l_in:self.X_sym}, A=self.A_sym, deterministic=True))
            self.f_gates.append(theano.function([self.X_sym, self.A_sym], self.gate_outputs[i], on_unused_input='warn'))

        

        
        parameters = lasagne.layers.get_all_params(self.l_out, trainable=True)
        updates = lasagne.updates.adam(self.train_loss, parameters, learning_rate=2e-3, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        self.f_train = theano.function([self.X_sym, self.train_y_sym, self.dev_y_sym, self.A_sym, self.train_indices_sym, self.dev_indices_sym], 
                                       [self.train_loss, self.train_acc, self.dev_loss, self.dev_acc, self.output], updates=updates, on_unused_input='warn')#, mode=theano.compile.MonitorMode(pre_func=inspect_inputs, post_func=inspect_outputs))
        self.f_val = theano.function([self.X_sym, self.A_sym, self.test_indices_sym], [self.test_pred, self.test_output], on_unused_input='warn')


        self.init_params = lasagne.layers.get_all_param_values(self.l_out)
        
        return self.l_out
    
    def fit(self, X, H, Y, train_indices, val_indices, n_epochs=10000, batch_size=1000, max_down=10, pseudolikelihood_thresh=0.2, verbose=True, seed=77):
        np.random.seed(seed)
        logging.info('training for {} epochs with batch size {}'.format(n_epochs, batch_size))
        best_params = None
        best_val_loss = sys.maxint
        best_val_acc = 0.0
        n_validation_down = 0
        report_k_epoch = 1

        X_train, y_train = X, Y[train_indices]
        y_dev = Y[val_indices]
        for n in xrange(n_epochs):
            l_train, acc_train, l_val, acc_val, all_probs = self.f_train(X_train, y_train, y_dev, H, train_indices, val_indices)
            l_train, acc_train = l_train.item(), acc_train.item()
            l_val, acc_val = l_val.item(), acc_val.item()

            if  l_val < best_val_loss:
                best_val_loss = l_val
                best_val_acc = acc_val
                best_params = lasagne.layers.get_all_param_values(self.l_out)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            if verbose:
                if n % report_k_epoch == 0:
                    logging.info('epoch {} train loss {:.2f} train acc {:.2f} val loss {:.2f} val acc {:.2f} best val acc {:.2f} maxdown {}'.format(n, l_train, acc_train, l_val, acc_val, best_val_acc, n_validation_down))
            if n_validation_down > max_down and n > 2 * report_k_epoch * max_down:
                logging.info('validation results went down. early stopping ...')
                break
        self.best_params = best_params
        lasagne.layers.set_all_param_values(self.l_out, best_params)        
        self.fitted = True
    
    def predict(self, X, A, test_indices):
        preds_test, prob_test = self.f_val(X, A, test_indices)
        return preds_test, prob_test
    
    def reset(self):
        lasagne.layers.set_all_param_values(self.l_out, self.init_params)

    def save(self, dumper, filename='./model.pkl'):
        if self.fitted:
            logging.info('dumping model params in {}'.format(filename))
            dumper(self.best_params, filename)
        else:
            logging.warn('The model is not trained yet!')
            
    def load(self, loader, filename):
         logging.info('loading the model from {}'.format(filename))
         self.best_params = loader(filename)
         lasagne.layers.set_all_param_values(self.l_out, self.best_params)
         self.fitted = True

    def get_gates(self, X, A):
        gates = []
        for fn in self.f_gates:
            gate = fn(X, A)
            gates.append(gate)
        return gates




        
if __name__ == '__main__':
    pass            
