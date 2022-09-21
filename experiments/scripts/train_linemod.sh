#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=2

python3 ./tools/train.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed 
  #--start_epoch 15 \
  #--resume_posenet pose_model_14_0.01099170069654631.pth