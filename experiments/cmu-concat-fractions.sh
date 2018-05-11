#!/bin/bash

for seed in 1 2 3 4 5
do
echo '******************** new seed ***************************'
echo $seed
    for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
    do
    THEANO_FLAGS='device=cuda2,optimizer=fast_run,floatX=float32' nice -n 9 python -u deepcca.py -hid 300 -bucket 50 -batch 300 -d ~/datasets/cmu/processed_data/  -reg 0.0 -dropout 0.5 -cel 5  -maxdown 10 -silent  -seed $seed -lblfraction $fraction
    done
done
exit 0
