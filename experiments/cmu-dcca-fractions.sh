#!/bin/bash

for seed in 1 2 3 4 5
do
echo '******************** new seed ***************************'
echo $seed
    for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
    do
    THEANO_FLAGS='device=cuda1,optimizer=fast_run,floatX=float32' nice -n 9 python -u deepcca.py -hid 300  -bucket 50 -batch 300 -d ~/datasets/cmu/processed_data/  -reg 2e-3 -dropout 0.5 -cel 5 -dcca -dccasize 500 -maxdown 10 -dccahid 1000 500 -dccareg 1e-4  -dccanonlin relu  -mlpbatchnorm -dccareload './data/deepcca-5685-2017-12-09 11:56:39.920863' -silent  -seed $seed -lblfraction $fraction
    done
done
exit 0
