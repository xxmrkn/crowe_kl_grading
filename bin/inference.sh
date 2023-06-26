#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=4
SIMG=''

mkdir -p ""

NODE=''

sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B user -B user ${SIMG} \
    python /src/inference.py \
    -m models=dense_reg \
    ++models.general.datalist=1,2 \
    ++models.inference.num_sampling=2"
