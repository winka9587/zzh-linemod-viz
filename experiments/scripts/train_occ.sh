#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/train_occ.py --dataset linemod\
    --dataset_root ./datasets/linemod/Linemod_preprocessed \
    --resume_posenet /data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_ape/pose_model_5_0.006174322933601659.pth