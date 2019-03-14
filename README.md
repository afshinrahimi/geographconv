Semi-supervised User Geolocation via Graph Convolutional Networks
=================================================================




Introduction
------------
This is a Theano implementation of Semi-supervised User Geolocation via Graph Convolutional Networks paper, published
in ACL 2018. It contains 3 geolocation models: 1) gcn, 2) deep cca, and 3) concat network and text.
The input data contains Twitter users with their tweets concatenated as a single document, and the
@-mentions in their tweets is used to build a graph between users.

The model uses the node features and the graph between them jointly to predict a location for users.
For more information about the models refer to the paper. It contains a Theano-based Graph Convolutional Network
model which can be used standalone for other experiments.


Geolocation Datasets
--------------------
Datasets are GEOTEXT a.k.a CMU (a small Twitter geolocation dataset)
and TwitterUS a.k.a NA (a bigger Twitter geolocation dataset) both
covering continental U.S. which can be downloaded from [here](https://www.amazon.com/clouddrive/share/kfl0TTPDkXuFqTZ17WJSnhXT0q6fGkTlOTOLZ9VVPNu)

If you want to use the preprocessed data e.g., X, A_hat in your own model download the pickle files from
[here] (https://www.amazon.com/clouddrive/share/yaH3HoyiKMVOrMxWLHeRaCEaCuH8MXoLz4UqUyOxCse) (1 *dump.pkl* file for each dataset).

Then load the file like this:

```python
import cPickle

def load_obj(filename, serializer=cPickle):
    with gzip.open(filename, 'rb') as fin:
        obj = serializer.load(fin)
    return obj


data = load_obj('dump.pkl')
A, X_train, Y_train, X_dev, Y_dev, X_test, Y_test, U_train, U_dev, U_test, classLatMedian, classLonMedian, userLocation = data
#A is the normalised laplacian matrix as A_hat in Kipf et al. (2016).
```
Then build your model and make predictions on X_test to get *y_pred*.
Then use the following function to evaluate the geolocation performance:

```python
from haversine import haversine
import logging
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

mean, median, acc, _, _ = geo_eval(y_true, y_pred, U_eval, classLatMedian, classLonMedian, userLocation)
```

Quick Start
-----------

1. Download the datasets and place them in ''./data/cmu'' and ''./data/na''

2. Create a new environment:

```conda create --name geo --file requirements.txt```

Activate the environment:

```conda activate geo```

Upgrade Theano and Lasagne:

```
pip install --upgrade https://github.com/Theano/Theano/archive/master.zip

pip install --upgrade https://github.com/Lasagne/Lasagne/archive/master.zip
```


2. To run the experiments, look at the experiments directory.

Note 1: The default parameters are not suitable for running the experiments.

Note 2: By changing the seed (e.g. using -seed 1 in command line) the results might slightly change, they might be slightly better or worse than the reported in the paper, but they shouldn't be very different.


For the GCN model:

CMU:

```
THEANO_FLAGS='device=cuda0,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 -bucket 50 -batch 500 -d ./data/cmu/ -enc latin1 -mindf 10 -reg 0.0 -dropout 0.5 -cel 5  -highway
```

NA:

```
THEANO_FLAGS='device=cpu,floatX=float32' python -u gcnmain.py -hid 600 600 600 -bucket 2400 -batch 500 -d ~/data/na/ -mindf 10 -reg 0.0 -dropout 0.5 -cel 15 -highway
```


WORLD:

```
THEANO_FLAGS='device=cpu,floatX=float32' python -u gcnmain.py -hid 900 900 900 -bucket 2400 -batch 500 -d ~/data/world/ -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -highway
```



Citation
--------
```
@InProceedings{rahimicontinuous2017,
  author    = {Rahimi, Afshin and Cohn, Trevor  and  Baldwin, Timothy},
  title     = {Semi-supervised User Geolocation via Graph Convolutional Networks},
  booktitle = {Proceedings of ACL2018},
  month     = {September},
  year      = {2018},
  address   = {Melbourne, Australia},
  publisher = {Association for Computational Linguistics},
  url       = {https://arxiv.org/abs/1804.08049}
}
```

Contact
-------
Afshin Rahimi <afshinrahimi@gmail.com>
