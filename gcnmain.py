#!/usr/bin/env python

"""A simple python script template.
"""

from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import os
import sys
import argparse
import pickle
import pdb
import copy
#from mlpconv import MLPCONV
#from mlp import MLP
import logging
import json
import numpy as np
from haversine import haversine
import gzip
import codecs
from collections import OrderedDict, defaultdict
import json
import re
import networkx as nx
import scipy as sp
from data import DataLoader, dump_obj, load_obj
import random
#import tensorflow as tf
import argparse
import sys
from collections import Counter
from gcnmodel import GraphConv


logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info('In order to work for big datasets fix https://github.com/Theano/Theano/pull/5721 should be applied to theano.')
np.random.seed(77)
model_args = None


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


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)
     

def preprocess_data(data_home, **kwargs):
    bucket_size = kwargs.get('bucket', 300)
    encoding = kwargs.get('encoding', 'utf-8')
    celebrity_threshold = kwargs.get('celebrity', 10)  
    mindf = kwargs.get('mindf', 10)
    dtype = kwargs.get('dtype', 'float32')
    one_hot_label = kwargs.get('onehot', False)
    vocab_file = os.path.join(data_home, 'vocab.pkl')
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
    vocab = dl.vectorizer.vocabulary_
    logging.info('saving vocab in {}'.format(vocab_file))
    dump_obj(vocab, vocab_file)
    logging.info('vocab dumped successfully!')
    U_test = dl.df_test.index.tolist()
    U_dev = dl.df_dev.index.tolist()
    U_train = dl.df_train.index.tolist()    

    dl.get_graph()  
    logging.info('creating adjacency matrix...')
    adj = nx.adjacency_matrix(dl.graph, nodelist=range(len(U_train + U_dev + U_test)), weight='w')
    
    adj.setdiag(0)
    #selfloop_value = np.asarray(adj.sum(axis=1)).reshape(-1,)
    selfloop_value = 1
    adj.setdiag(selfloop_value)
    n,m = adj.shape
    diags = adj.sum(axis=1).flatten()
    with sp.errstate(divide='ignore'):
        diags_sqrt = 1.0/sp.sqrt(diags)
    diags_sqrt[sp.isinf(diags_sqrt)] = 0
    D_pow_neghalf = sp.sparse.spdiags(diags_sqrt, [0], m, n, format='csr')
    A = D_pow_neghalf * adj * D_pow_neghalf
    A = A.astype(dtype)
    logging.info('adjacency matrix created.')

    X_train = dl.X_train
    X_dev = dl.X_dev
    X_test = dl.X_test
    Y_test = dl.test_classes
    Y_train = dl.train_classes
    Y_dev = dl.dev_classes
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
    
    data = (A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation)
    if not model_args.builddata:
        logging.info('dumping data in {} ...'.format(str(dump_file)))
        dump_obj(data, dump_file)
        logging.info('data dump finished!')

    return data


def main(data, args, **kwargs):
    batch_size = kwargs.get('batch', 500)
    hidden_size = kwargs.get('hidden', [100])
    dropout = kwargs.get('dropout', 0.0)
    regul = kwargs.get('regularization', 1e-6)
    dtype = 'float32'
    dtypeint = 'int32'
    check_percentiles = kwargs.get('percent', False)
    A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
    logging.info('stacking training, dev and test features and creating indices...')
    X = sp.sparse.vstack([X_train, X_dev, X_test])
    if len(Y_train.shape) == 1:
        Y = np.hstack((Y_train, Y_dev, Y_test))
    else:
        Y = np.vstack((Y_train, Y_dev, Y_test))
    Y = Y.astype(dtypeint)
    X = X.astype(dtype)
    A = A.astype(dtype)
    if args.vis:
        from deepcca import draw_representations
        draw_representations(A.dot(X), Y, filename='gconv1.pdf')
        draw_representations(A.dot(A.dot(X)), Y, filename='gconv2.pdf')
    input_size = X.shape[1]
    output_size = np.max(Y) + 1
    verbose = not args.silent
    fractions = args.lblfraction
    stratified = False
    all_train_indices = np.asarray(range(0, X_train.shape[0])).astype(dtypeint)
    logging.info('running mlp with graph conv...')
    clf = GraphConv(input_size=input_size, output_size=output_size, hid_size_list=hidden_size, regul_coef=regul, drop_out=dropout, batchnorm=args.batchnorm, highway=model_args.highway)
    clf.build_model(A, use_text=args.notxt, use_labels=args.lp, seed=model_args.seed)

    for percentile in fractions:
        logging.info('***********percentile %f ******************' %percentile)
        model_file = './data/model-{}-{}.pkl'.format(A.shape[0], percentile)
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
        #train_indices = np.asarray(range(0, int(percentile * X_train.shape[0]))).astype(dtypeint)
        dev_indices = np.asarray(range(X_train.shape[0], X_train.shape[0] + X_dev.shape[0])).astype(dtypeint)
        test_indices = np.asarray(range(X_train.shape[0] + X_dev.shape[0], X_train.shape[0] + X_dev.shape[0] + X_test.shape[0])).astype(dtypeint)
        # do not train, load
        if args.load:
            report_results = False
            clf.load(load_obj, model_file)
        else:
            #reset the network parameters if already fitted with another data
            if clf.fitted:
                clf.reset()
            clf.fit(X, A, Y, train_indices=train_indices, val_indices=dev_indices, n_epochs=10000, batch_size=batch_size, max_down=args.maxdown, verbose=verbose, seed=model_args.seed)
            if args.save:
                clf.save(dump_obj, model_file)

            logging.info('dev results:')
            y_pred, _ = clf.predict(X, A, dev_indices)
            mean, median, acc, distances, latlon_true, latlon_pred = geo_eval(Y_dev, y_pred, U_dev, classLatMedian, classLonMedian, userLocation)
            with open('gcn_{}_percent_pred_{}.pkl'.format(percentile, output_size), 'wb') as fout:
                pickle.dump((distances, latlon_true, latlon_pred), fout)
            logging.info('test results:')
            y_pred, _ = clf.predict(X, A, test_indices)
            geo_eval(Y_test, y_pred, U_test, classLatMedian, classLonMedian, userLocation)

    if args.feature_report:
        vocab_file = os.path.join(args.dir, 'vocab.pkl')
        if not os.path.exists(vocab_file):
            logging.error('vocab file {} not found'.format(vocab_file))
            return
        else:
            vocab = load_obj(vocab_file)
        logging.info('{} vocab loaded from file'.format(len(vocab)))
        train_vocab = set([term for term, count in Counter(np.nonzero(X[train_indices])[1]).iteritems() if count >= 10])
        dev_vocab = set(np.nonzero(X[dev_indices].sum(axis=0))[1])
        X_onehot = sp.sparse.diags([1] * len(vocab), dtype=dtype)
        A_onehot = X_onehot
        feature_report(clf, vocab, X_onehot, A_onehot, classLatMedian, classLonMedian, train_vocab, dev_vocab, topk=200, dtypeint=dtypeint)


def feature_report(model, vocab, X, A, classLatMedian, classLonMedian, train_vocab=set(), dev_vocab=set(), topk=20, dtypeint='int32', filename='important_features.txt'):
    eval_indices = np.asarray(range(X.shape[0])).astype(dtypeint)
    preds, probs = model.predict(X, A, eval_indices)
    id2v = {v: k for k, v in vocab.iteritems()}
    logging.info('{} train vocab are being excluded!'.format(len(train_vocab)))
    #select top k most important features for each class
    feature_importance = np.argsort(-probs, axis=0)
    with codecs.open(filename, 'w', encoding='utf-8') as fout:
        for lbl in range(probs.shape[1]):
            important_vocab = ' '.join([id2v[idx] for idx in feature_importance[:, lbl].reshape(-1).tolist() if idx not in train_vocab][0:topk])
            lat, lon = classLatMedian[str(lbl)], classLonMedian[str(lbl)]
            fout.write(u'location: {},{} \nimportant features: {} \n\n'.format(lat, lon, important_vocab))
    logging.info('important features are written to {}'.format(filename))


def parse_args(argv):
    """
    Parse commandline arguments.
    Arguments:
        argv -- An argument list without the program name.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument( '-i','--dataset', metavar='str', help='dataset for dialectology', type=str, default='na')
    parser.add_argument( '-bucket','--bucket', metavar='int', help='discretisation bucket size', type=int, default=300)
    parser.add_argument( '-batch','--batch', metavar='int', help='SGD batch size', type=int, default=500)
    parser.add_argument('-hid', nargs='+', type=int, help="list of hidden layer sizes", default=[100])
    parser.add_argument( '-mindf','--mindf', metavar='int', help='minimum document frequency in BoW', type=int, default=10)
    parser.add_argument( '-d','--dir', metavar='str', help='home directory', type=str, default='./data')
    parser.add_argument( '-enc','--encoding', metavar='str', help='Data Encoding (e.g. latin1, utf-8)', type=str, default='utf-8')
    parser.add_argument( '-reg','--regularization', metavar='float', help='regularization coefficient)', type=float, default=1e-6)
    parser.add_argument( '-cel','--celebrity', metavar='int', help='celebrity threshold', type=int, default=10)
    parser.add_argument( '-conv', '--convolution', action='store_true', help='if true do convolution')
    parser.add_argument( '-tune', '--tune', action='store_true', help='if true tune the hyper-parameters')
    parser.add_argument( '-tf', '--tensorflow', action='store_true', help='if exists run with tensorflow')
    parser.add_argument( '-batchnorm', action='store_true', help='if exists do batch normalization')
    parser.add_argument('-dropout', type=float, help="dropout value default(0)", default=0)
    parser.add_argument( '-percent', action='store_true', help='if exists loop over different train/dev proportions')
    parser.add_argument( '-vis', metavar='str', help='visualise representations', type=str, default=None)
    parser.add_argument('-builddata', action='store_true', help='if exists do not reload dumped data, build it from scratch')
    parser.add_argument('-lp', action='store_true', help='if exists use label information')
    parser.add_argument('-notxt', action='store_false', help='if exists do not use text information')
    parser.add_argument( '-maxdown', help='max iter for early stopping', type=int, default=10)
    parser.add_argument('-silent', action='store_true', help='if exists be silent during training')
    parser.add_argument('-highway', action='store_true', help='if exists use highway connections else do not')
    parser.add_argument( '-seed', metavar='int', help='random seed', type=int, default=77)
    parser.add_argument('-save', action='store_true', help='if exists save the model after training')
    parser.add_argument('-load', action='store_true', help='if exists load pretrained model from file')
    parser.add_argument('-feature_report', action='store_true', help='if exists report the important features of each location')
    parser.add_argument('-lblfraction', nargs='+', type=float, help="fraction of labelled data used for training e.g. 0.01 0.1", default=[1.0])
    args = parser.parse_args(argv)
    return args


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    model_args = args

    data = preprocess_data(data_home=args.dir, encoding=args.encoding, celebrity=args.celebrity, bucket=args.bucket, mindf=args.mindf)
    main(data, args, batch=args.batch, hidden=args.hid, regularization=args.regularization, dropout=args.dropout, percent=args.percent)
        
    
