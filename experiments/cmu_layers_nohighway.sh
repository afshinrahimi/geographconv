#!/bin/bash
for seed in 1 2 3 4 5
do
#different number of layers, without highway
echo $seed
echo '******************** new seed ***************'
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300     -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300     -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300    -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300    -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u resnet.py -hid 300 300 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -silent

done 
exit 0
