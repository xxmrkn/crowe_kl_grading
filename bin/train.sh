#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=4
SIMG=''

mkdir -p ""

NODE=''

sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B user -B user ${SIMG} \
    python src/train.py \
    -m models=vit_cls,vgg_cls,dense_cls \
    ++models.general.datalist=2 \
    ++models.optimizer.wd=0.001"
