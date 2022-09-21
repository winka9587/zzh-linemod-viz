#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/train.py --dataset ycb\
  --dataset_root /data1/zzh/YCB_Video_Dataset/