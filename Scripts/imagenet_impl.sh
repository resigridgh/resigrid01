#!/bin/bash

cd ~/resigrid01 || exit 1
export PYTHONPATH="$(pwd)/src:$PYTHONPATH"

nohup python Scripts/imagenet_impl.py \
  --epochs 10000 \
  --train_ratio 0.005 \
  --val_ratio 0.005 \
  > training.log 2>&1 &
