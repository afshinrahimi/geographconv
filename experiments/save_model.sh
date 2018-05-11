#!/bin/bash
THEANO_FLAGS='device=cpu,floatX=float32' python -u gcnmain.py -hid 600 600 600 -bucket 2400 -batch 500 -d ~/datasets/na/processed_data/ -mindf 10 -reg 0.0 -dropout 0.5 -cel 15 -highway -save
