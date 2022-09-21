#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/eval_linemod_render.py --dataset_root ./datasets/linemod/Linemod_preprocessed\
  --model /data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_cam/pose_model_5_0.007620463889279385.pth\
  --refine_model /data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_cam/pose_refine_model_16_0.004853710358507028.pth