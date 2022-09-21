#!/bin/bash

set -x
set -e

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=4

python3 ./tools/eval_occ.py --dataset_root /data1/zzh/OCCLUSION_LINEMOD\
  --model /data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_ape/pose_model_5_0.006174322933601659.pth\
  --refine_model /data1/zzh/DenseFusion-Pytorch-1.0/trained_models/linemod_ape/pose_refine_model_11_0.00457129196823752.pth