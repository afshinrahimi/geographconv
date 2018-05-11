#!/bin/bash
for seed in 1 2 3 4 5
do
#THEANO_FLAGS='device=cuda3,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300    -bucket $bucket -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -builddata
echo $seed
echo '******************** new seed ***************'
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300     -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent 
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300     -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300    -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300    -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent

THEANO_FLAGS='device=cpu,floatX=float32' nice -n 9 python -u gcnmain.py -hid 300 300 300 300 300 300 300 300 300 300 300   -bucket 50 -batch 500 -d ~/datasets/cmu/processed_data/  -mindf 10 -reg 0.0 -dropout 0.5 -cel 5 -seed $seed -highway -silent
done 
exit 0
