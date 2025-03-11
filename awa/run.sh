#!/bin/bash

set -x 

python eval_gpsr.py
python train.py
python eval.py --epoch=all
python eval_compare.py