#!/bin/bash

for seed in 1 2 3 4 5
do
echo '******************** new seed ***************************'
echo $seed
    for fraction in 0.01 0.02 0.05 0.1 0.2 0.4 0.6
    do
    THEANO_FLAGS='device=cuda0,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300    -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -silent -highway  -seed $seed -lblfraction $fraction
    done
done
exit 0
