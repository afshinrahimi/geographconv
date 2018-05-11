'''
Created on 26 Sep. 2017

@author: af
'''
import matplotlib
from sklearn.preprocessing.data import StandardScaler
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import seaborn
seaborn.set_style('white')
from collections import Counter
import sys
import numpy as np
import argparse
import pdb
import pickle
import cPickle
import hickle
import gzip
import os
from datetime import datetime
from haversine import haversine
import theano
import theano.tensor as T
import lasagne
import networkx as nx
import scipy as sp
from lasagne.regularization import l1, l2
import theano.sparse as S
from lasagne.layers import DenseLayer, DropoutLayer
import logging
from data import DataLoader, dump_obj, load_obj
from mlp import MLPDense



logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
model_args = None

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

def geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation):
    assert len(y_pred) == len(U_eval), "#preds: %d, #users: %d" %(len(y_pred), len(U_eval))
    distances = []
    latlon_pred = []
    latlon_true = []
    for i in range(0, len(y_pred)):
        user = U_eval[i]
        location = userLocation[user].split(',')
        lat, lon = float(location[0]), float(location[1])
        latlon_true.append([lat, lon])
        prediction = str(y_pred[i])
        lat_pred, lon_pred = classLatMedian[prediction], classLonMedian[prediction]
        latlon_pred.append([lat_pred, lon_pred])  
        distance = haversine((lat, lon), (lat_pred, lon_pred))
        distances.append(distance)

    acc_at_161 = 100 * len([d for d in distances if d < 161]) / float(len(distances))

    logging.info( "Mean: " + str(int(np.mean(distances))) + " Median: " + str(int(np.median(distances))) + " Acc@161: " + str(int(acc_at_161)))
        
    return np.mean(distances), np.median(distances), acc_at_161, distances, latlon_true, latlon_pred

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



def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    dump_file = os.path.join(data_home, 'dump.pkl')
    if os.path.exists(dump_file) and not model_args.builddata:
        logging.info('loading data from dumped file...')
        data = load_obj(dump_file)
        logging.info('loading data finished!')
        return data
    dl = DataLoader(data_home=data_home, bucket_size=bucket_size, encoding=encoding, 
                    celebrity_threshold=celebrity_threshold, one_hot_labels=one_hot_label, mindf=mindf, token_pattern=r'(?u)(?<![@])#?\b\w\w+\b')
    dl.load_data()
    dl.assignClasses()
    dl.tfidf()
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()    

    dl.get_graph()  
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=xrange(len(U_train + U_dev + U_test)), weight='w')
    #adj[adj > 0] = 1
    adj.setdiag(1)
    n,m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0/sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    H = D_pow_neghalf * adj * D_pow_neghalf
    H = H.astype(dtype)
    logging.info('adjacency matrix created.')
    
    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes.astype('int32')
    Y_train = dl.train_classes.astype('int32')
    Y_dev = dl.dev_classes.astype('int32')
    classLatMedian = {str(c):dl.cluster_median[c][0] for c in dl.cluster_median}
    classLonMedian = {str(c):dl.cluster_median[c][1] for c in dl.cluster_median}
    
    
    P_test = [str(a[0]) + ',' + str(a[1]) for a in dl.df_test[['lat', 'lon']].values.tolist()]
    P_train = [str(a[0]) + ',' + str(a[1]) for a in dl.df_train[['lat', 'lon']].values.tolist()]
    P_dev = [str(a[0]) + ',' + str(a[1]) for a in dl.df_dev[['lat', 'lon']].values.tolist()]
    userLocation = {}
    for i, u in enumerate(U_train):
        userLocation[u] = P_train[i]
    for i, u in enumerate(U_test):
        userLocation[u] = P_test[i]
    for i, u in enumerate(U_dev):
        userLocation[u] = P_dev[i]
    
    data = (H, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')
    return data


        

class DeepCCA():
    """
    A simple multilayer-perceptron class
    """
    def __init__(self):
        # CCA regularization 
        self.reg1 = 1e-4
        self.reg2 = 1e-4

        # for numerical statibility (from H. Kamper via W. Wang)
        self.eps = 1e-12
    
    #copied from https://github.com/msamribeiro/deep-cca/blob/master/model/cca_layer.py
    def cca_loss(self, data1, data2, cca_dim=None):

        n_data = data1.shape[0]
        in_dim1 = data1.shape[1]
        in_dim2 = data2.shape[1]
        cca_dim = cca_dim if cca_dim else min(in_dim1, in_dim2)

        # center the data
        data1 -= T.mean(data1, axis=0)
        data2 -= T.mean(data2, axis=0)
        data1 = data1.T
        data2 = data2.T

        # find covariance matrices
        sigma11 = (1/(n_data-1.)) * T.dot(data1, data1.T)
        sigma22 = (1/(n_data-1.)) * T.dot(data2, data2.T)
        sigma12 = (1/(n_data-1.)) * T.dot(data1, data2.T)

        # add regulatization
        sigma11 += self.reg1 * T.eye(in_dim1)
        sigma22 += self.reg2 * T.eye(in_dim2)

        # diagonalize covariance matrices to find inverses
        diag1, q1 = T.nlinalg.eigh(sigma11)
        diag2, q2 = T.nlinalg.eigh(sigma22)

        # numerical stability (from H. Kamper, via W. Wang)
        # http://stackoverflow.com/questions/20590909/returning-the-index-of-a-value-in-theano-vector
        idx = T.gt(diag1, self.eps).nonzero()[0] 
        diag1 = diag1[idx]
        q1 = q1[:, idx]
        idx = T.gt(diag2, self.eps).nonzero()[0]
        diag2 = diag2[idx]
        q2 = q2[:, idx]

        # find correlation matrix T
        sigma11_inv = T.dot(q1, T.dot(T.diag(diag1**(-0.5)), q1.T))
        sigma22_inv = T.dot(q2, T.dot(T.diag(diag2**(-0.5)), q2.T))
        T_corr = T.dot(sigma11_inv, T.dot(sigma12, sigma22_inv))

        # find the singular values of T through the eigenvalues of TT.T
        Tdiag, Tevec = T.nlinalg.eigh(T.dot(T_corr, T_corr.T))
        Tdiag = Tdiag[T.gt(Tdiag, self.eps).nonzero()[0]]
        Tdiag.sort()
        Tdiag = Tdiag[::-1]**(0.5)

        # take the top k canonical components (top k singular values)
        # here we negate corr to treat this as a minimization problem
        corr = -T.sum(Tdiag[:cca_dim])
        mean = -T.mean(Tdiag[:cca_dim])

        return corr , mean

    
    def build(self, v1_input_size, v2_input_size, architecture, regul_coef=0.0, dropout=0.0, lr=1e-3, batchnorm=False, seed=77):
        np.random.seed(seed)
        V1_sym = S.csr_matrix(name='view1', dtype='float32')
        V2_sym = S.csr_matrix(name='view2', dtype='float32')
        self.batchnorm = batchnorm
        logging.info('building deepcca network with batchnorm {} regul {} lr {} layers {} cca_dim {}'.format(str(self.batchnorm), regul_coef, lr, str(architecture), model_args.dccasize))
        l_out_view1 = self.build_mlp(V1_sym, input_size=v1_input_size, architecture=architecture, dropout=dropout)
        l_out_view2 = self.build_mlp(V2_sym, input_size=v2_input_size, architecture=architecture, dropout=dropout)
        self.l_out_view1 = l_out_view1
        self.l_out_view2 = l_out_view2
        output_view1 = lasagne.layers.get_output(l_out_view1)
        output_view2 = lasagne.layers.get_output(l_out_view2)
        loss_cca, _ = self.cca_loss(output_view1, output_view2, cca_dim=model_args.dccasize)
                
        regul_loss1 = lasagne.regularization.regularize_network_params(l_out_view1, penalty=l2)
        regul_loss2 = lasagne.regularization.regularize_network_params(l_out_view2, penalty=l2)
        regul_loss = (regul_loss1 + regul_loss2) * regul_coef

        loss = loss_cca + regul_loss
        params = lasagne.layers.get_all_params(l_out_view1, trainable=True) + lasagne.layers.get_all_params(self.l_out_view2, trainable=True)

        updates = lasagne.updates.adam(loss, params, learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8)
        
        #updates = lasagne.updates.sgd(loss, params, learning_rate=lr)
        self.f_train = theano.function([V1_sym, V2_sym], loss_cca, updates=updates)
        self.f_val = theano.function([V1_sym, V2_sym], loss_cca)
        self.f_predict = theano.function([V1_sym, V2_sym], [output_view1, output_view2, loss_cca])

                
    def build_mlp(self, X_sym, input_size, architecture, dropout=0.0): 
        l_in = lasagne.layers.InputLayer(shape=(None, input_size), input_var=X_sym)
        l_hid = l_in
        def dcca_activation(x):
            #inverse of x^3/3 + x
            a = ((3 * x + (9 * x **2 + 4) ** 0.5) ** (1.0/3)) / 2 ** (1.0/3)
            y =  a - 1.0/a
            return y
        num_layers = len(architecture)
        activation = lasagne.nonlinearities.sigmoid
        W = lasagne.init.GlorotUniform()
        if model_args.dccanonlin == 'custom':
            activation = dcca_activation
        elif model_args.dccanonlin == 'tanh':
            activation = lasagne.nonlinearities.tanh
        elif model_args.dccanonlin == 'relu':
            activation = lasagne.nonlinearities.rectify
            W = lasagne.init.GlorotUniform(gain='relu')
        
        
        for i, h_size in enumerate(architecture):
            if i == 0:
                #incoming is sparse
                l_hid = SparseInputDenseLayer(l_hid, num_units=h_size, nonlinearity=activation, W=W)
                if self.batchnorm:
                    l_hid = lasagne.layers.batch_norm(l_hid)
                #l_hid = lasagne.layers.dropout(l_hid, p=dropout) 
            elif i < num_layers - 1:
                l_hid = DenseLayer(l_hid, num_units=h_size, nonlinearity=activation, W=W)
                if self.batchnorm:
                    l_hid = lasagne.layers.batch_norm(l_hid)
            else:
                #output is linear
                l_out = DenseLayer(l_hid, num_units=h_size, nonlinearity=lasagne.nonlinearities.linear)
                

        return l_out
    
    def fit(self, V1, V2, train_indices, val_indices, test_indices, n_epochs=10, early_stopping_max_down=3, batch_size=1000):
        best_params1 = None
        best_params2 = None
        best_val_loss = sys.maxint
        n_validation_down = 0
        V1_train = V1[train_indices, :]
        V2_train = V2[train_indices, :]
        V1_dev = V1[val_indices, :] 
        V2_dev = V2[val_indices, :]
        logging.info('training with batch size {}'.format(batch_size))
        for n in xrange(n_epochs):
            l_train = []
            for batch in iterate_minibatches(V1_train, V2_train, batch_size, shuffle=False):
                l_train.append(self.f_train(batch[0], batch[1]))
            l_train = np.mean(l_train)
            l_val = self.f_val(V1_dev, V2_dev).item()
            #after k iterations improvement should be higher than 0.1
            k = 100
            improvement = 1.0
            if (l_val < best_val_loss and n < k) or (l_val < best_val_loss - improvement):
                best_val_loss = l_val
                best_params1 = lasagne.layers.get_all_param_values(self.l_out_view1)
                best_params2 = lasagne.layers.get_all_param_values(self.l_out_view2)
                n_validation_down = 0
            else:
                #early stopping
                n_validation_down += 1
            logging.info('epoch {} train loss {:.2f} val loss {:.2f} numdown {}'.format(n, l_train, l_val, n_validation_down))
            if n_validation_down > early_stopping_max_down:
                logging.info('validation results went down. early stopping ...')
                break
        
        lasagne.layers.set_all_param_values(self.l_out_view1, best_params1)
        lasagne.layers.set_all_param_values(self.l_out_view2, best_params2)
        
        logging.info('***************** final results based on best validation **************')
        V1_test, V2_test, l_test = self.f_predict(V1[test_indices], V2[test_indices])
        logging.info('test loss:{}'.format(l_test))
        filename = 'deepcca-{}-{}'.format(train_indices.shape[0], str(datetime.now()))
        logging.info('dumping deepcca params in {} '.format(filename))
        if V1.shape[0] > 1000000:
            dump_obj((str(model_args), best_params1, best_params2), filename, serializer=hickle)
        else:
            dump_obj((model_args, best_params1, best_params2), filename)
    def set_params(self, best_params1, best_params2):
        lasagne.layers.set_all_param_values(self.l_out_view1, best_params1)
        lasagne.layers.set_all_param_values(self.l_out_view2, best_params2)


def linear_cca(H1, H2, outdim_size):
    """
    copied from https://github.com/VahidooX/DeepCCA/blob/master/linear_cca.py
    An implementation of linear CCA
    # Arguments:
        H1 and H2: the matrices containing the data for view 1 and view 2. Each row is a sample.
        outdim_size: specifies the number of new features
    # Returns
        A and B: the linear transformation matrices 
        mean1 and mean2: the means of data for both views
    """
    r1 = 1e-4
    r2 = 1e-4

    m = H1.shape[0]
    o = H1.shape[1]

    mean1 = np.mean(H1, axis=0)
    mean2 = np.mean(H2, axis=0)
    H1bar = H1 - np.tile(mean1, (m, 1))
    H2bar = H2 - np.tile(mean2, (m, 1))

    SigmaHat12 = (1.0 / (m - 1)) * np.dot(H1bar.T, H2bar)
    SigmaHat11 = (1.0 / (m - 1)) * np.dot(H1bar.T, H1bar) + r1 * np.identity(o)
    SigmaHat22 = (1.0 / (m - 1)) * np.dot(H2bar.T, H2bar) + r2 * np.identity(o)

    [D1, V1] = np.linalg.eigh(SigmaHat11)
    [D2, V2] = np.linalg.eigh(SigmaHat22)
    SigmaHat11RootInv = np.dot(np.dot(V1, np.diag(D1 ** -0.5)), V1.T)
    SigmaHat22RootInv = np.dot(np.dot(V2, np.diag(D2 ** -0.5)), V2.T)

    Tval = np.dot(np.dot(SigmaHat11RootInv, SigmaHat12), SigmaHat22RootInv)

    [U, D, V] = np.linalg.svd(Tval)
    V = V.T
    A = np.dot(SigmaHat11RootInv, U[:, 0:outdim_size])
    B = np.dot(SigmaHat22RootInv, V[:, 0:outdim_size])
    D = D[0:outdim_size]
    return A, B, mean1, mean2

def draw_representations(X, y, k=4, do_pca=True, filename='output.pdf'):
    class_count = Counter(y.tolist()).most_common(k)
    num_samples =  class_count[3][1] - class_count[3][1] % 10 
    all_lbls = []
    all_samples = []
    for i, cc in enumerate(class_count):
        lbl, _ = cc
        samples = X[y == lbl][0:num_samples, :]
        samples = samples.todense() if sp.sparse.issparse(samples) else samples
        lbls = y[y == lbl][0:num_samples]
        lbls[:] = i
        all_samples.append(samples)
        all_lbls.append(lbls)
    all_lbls = np.hstack(all_lbls)
    all_samples = np.vstack(all_samples)
    if do_pca:
        pca = PCA(n_components=50, random_state=model_args.seed)
        all_samples = pca.fit_transform(all_samples)
    tsne = TSNE(n_components=2, random_state=model_args.seed)
    embeddings = tsne.fit_transform(all_samples)    
    chosen_indices = np.random.choice(np.arange(embeddings.shape[0]), size = k * min(50, num_samples), replace=False)
    chosen_embeddings = embeddings[chosen_indices, :]
    chosen_ys = all_lbls[chosen_indices]
    axes = plt.gca()
    #axes.set_xlim([-8,8])
    #axes.set_ylim([-8,7])
    #axes.set_xlim([-3.5,5.5])
    #axes.set_ylim([-7,2])
    axes.set_xlim([-20,20])
    axes.set_ylim([-20,20])
    
    plt.axis('off')
    plt.scatter(chosen_embeddings[:, 0], chosen_embeddings[:, 1], c=chosen_ys, cmap=plt.cm.get_cmap("Set1", k))
    plt.savefig(filename)
    plt.close()
    #pdb.set_trace()


def main(data, args, **kwargs):
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', [100])
    dropout = kwargs.get('dropout', 0.0)
    regul = kwargs.get('regularization', 1e-6)
    dtype = 'float32'
    dtypeint = 'int32'
    check_percentiles = kwargs.get('percent', False)
    H, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    Y_dev = Y_dev.astype(dtypeint)
    Y_test = Y_test.astype(dtypeint)
    logging.info('stacking training, dev and test features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))
    Y = Y.astype('int32')
    X = X.astype(dtype)
    H = H.astype(dtype)
    input_size = X.shape[1]
    output_size = np.max(Y) + 1
    

    train_indices = np.asarray(range(0, X_train.shape[0])).astype('int32')
    dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype('int32')
    test_indices = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype('int32')
    batch_size = min(batch_size, train_indices.shape[0])
    if args.dcca:
        logging.info('running deepcca...')
        deepcca = DeepCCA()
        deepcca.build(X.shape[1], H.shape[1], architecture=args.dccahid, regul_coef=args.dccareg, dropout=dropout, lr=args.dccalr, batchnorm=args.dccabatchnorm, seed=model_args.seed)
        if args.dccareload:
            #for the big dataset use pickle instead of cPickle
            if X.shape[0] > 1000000:
                loaded_args, params1, params2 = load_obj(args.dccareload, serializer=hickle)
            else:
                loaded_args, params1, params2 = load_obj(args.dccareload)
            logging.info(loaded_args)
            deepcca.set_params(params1, params2)
            
        else:
            deepcca.fit(V1=X, V2=H, train_indices=train_indices, val_indices=dev_indices, test_indices=test_indices, n_epochs=500, 
                        early_stopping_max_down=args.maxdown, batch_size=train_indices.shape[0])
        V1_cca, V2_cca, l_cca = deepcca.f_predict(X, H)
        should_run_cca_on_outputs = True
        if should_run_cca_on_outputs:
            #run linear cca on the outputs of mlp
            A, B, mean1, mean2 = linear_cca(V1_cca, V2_cca, outdim_size=args.dccasize)
            V1_cca = V1_cca - mean1
            V2_cca = V2_cca - mean2
            V1_cca = np.dot(V1_cca, A)
            V2_cca = np.dot(V2_cca, B)
        X_cca = np.hstack((V1_cca, V2_cca)).astype(dtype)
    else:
        logging.info('No shared deepcca representation, just concatenation!')
        X_cca = sp.sparse.hstack([X, H]).astype(dtype).tocsr()
    stratified = False
    all_train_indices = train_indices
    fractions = args.lblfraction
    clf = MLPDense(input_sparse=sp.sparse.issparse(X_cca), in_size=X_cca.shape[1], out_size=output_size,
                   architecture=hidden_size, regul=regul, dropout=dropout, lr=args.mlplr,
                   batchnorm=args.mlpbatchnorm)
    clf.build(seed=model_args.seed)

    for percentile in fractions:
        logging.info('***********percentile %f ******************' %percentile)
        if stratified:
            all_chosen = []
            for lbl in range(0, np.max(Y_train) + 1): 
                lbl_indices = all_train_indices[Y_train == lbl]
                selection_size =  int(percentile * len(lbl_indices)) + 1
                lbl_chosen = np.random.choice(lbl_indices, size=selection_size, replace=False).astype(dtypeint)
                all_chosen.append(lbl_chosen)
            train_indices = np.hstack(all_chosen) 
        else:
            selection_size = min(int(percentile * X.shape[0]), all_train_indices.shape[0])
            train_indices = np.random.choice(all_train_indices, size=selection_size, replace=False).astype(dtypeint)
        num_training_samples = train_indices.shape[0]
        logging.info('{} training samples'.format(num_training_samples))
        X_train = X_cca[train_indices, :]
        Y_train_chosen = Y_train[train_indices].astype('int32')
        X_dev = X_cca[dev_indices, :]
        X_test = X_cca[test_indices, :]
        if args.vis:
            draw_representations(X_train, Y_train_chosen, k=4, do_pca=True, filename=args.vis)
        if clf.fitted:
            clf.reset()
        clf.fit(X_train, Y_train_chosen, X_dev, Y_dev, n_epochs=1000, early_stopping_max_down=args.maxdown, verbose=not args.silent, batch_size=min(batch_size, train_indices.shape[0]), seed=model_args.seed)
        dev_pred = clf.predict(X_dev)
        test_pred = clf.predict(X_test)
        logging.info('Dev predictions')
        mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(Y_dev, dev_pred, U_dev, classLatMedian, classLonMedian, userLocation) 
        with open('dcca_{}_percent_pred_{}.pkl'.format(percentile, output_size) if args.dcca else 'concat_{}_percent_pred_{}.pkl'.format(percentile, output_size), 'wb') as fout:
            pickle.dump((distances, latlon_true, latlon_pred), fout)
        logging.info('Test predictions')
        geo_eval(Y_test, test_pred, U_test, classLatMedian, classLonMedian, userLocation)
         

def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '-bucket','--bucket', metavar='int', help='discretisation bucket size', type=int, default=2400)
    parser.add_argument( '-batch','--batch', metavar='int', help='SGD batch size', type=int, default=500)
    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument( '-mindf','--mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float', help='regularization coefficient)', type=float, default=1e-4)
    parser.add_argument( '-cel','--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument( '-tune', '--tune', action='store_true', help='if true tune the hyper-parameters')
    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)
    parser.add_argument( '-percent', action='store_true', help='if exists loop over different train/dev proportions')
    parser.add_argument( '-dcca', action='store_true', help='if exists run dcca over text and network views')
    parser.add_argument( '-maxdown', metavar='int', help='early stopping max down', type=int, default=5)
    parser.add_argument( '-dccasize', metavar='int', help='DCCA output size for each view', type=int, default=300)
    parser.add_argument('-dccahid', nargs='+', type=int, help="list of dcca hidden layer sizes", default=[600, 600])
    parser.add_argument( '-dccareg', metavar='float', help='l2 regularization coefficient for dcca', type=float, default=1e-4)
    parser.add_argument( '-dccalr', metavar='float', help='learning rate for dcca', type=float, default=3e-4)
    parser.add_argument( '-mlplr', metavar='float', help='learning rate for mlp', type=float, default=3e-4)
    parser.add_argument( '-dccareload', metavar='str', help='deepcca parameter reload', type=str, default=None)
    parser.add_argument( '-vis', metavar='str', help='embedding vis output file e.g. output.pdf', type=str, default=None)
    parser.add_argument( '-dccabatchnorm', action='store_true', help='if exists do batch norm')
    parser.add_argument( '-mlpbatchnorm', action='store_true', help='if exists do batch norm')
    parser.add_argument( '-dccanonlin', metavar='str', help='nonlinearity of dcca', type=str, default='relu')
    parser.add_argument('-builddata', action='store_true', help='if exists do not reload dumped data, build it from scratch')
    parser.add_argument('-seed', metavar='int' , help='random seed', type=int, default=77)
    parser.add_argument('-lblfraction', nargs='+', type=float, help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    parser.add_argument('-silent', action='store_true', help='if exists be silent during training')


    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args
    assert args.dccasize <= args.dccahid[-1]
    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
    main(data, args, batch=args.batch, hidden=args.hid, regularization=args.regularization, dropout=args.dropout, percent=args.percent)
