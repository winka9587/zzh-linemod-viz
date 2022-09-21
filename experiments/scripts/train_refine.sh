#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/train_refine.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed \
  --resume_posenet pose_model_30_0.008178278177831611.pth