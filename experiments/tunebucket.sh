#!/bin/bash

for bucket in 16 32 64 128 256 512
do
for seed in 1 2 3 4 5
do
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 600 600 600    -bucket $bucket -batch 500 -d ~/datasets/na/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 15 -builddata -highway -seed $seed
done
done
exit 0
