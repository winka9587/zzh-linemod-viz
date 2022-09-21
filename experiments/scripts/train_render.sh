#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/train_render.py --dataset linemod\
  --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --start_epoch 3\
  --resume_posenet pose_model_2_0.00810950485220377.pth
