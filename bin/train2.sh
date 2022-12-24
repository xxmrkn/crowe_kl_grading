#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=2
SIMG=''

mkdir -p "${PROJECT}/slurm"

NODE='cl-panda'
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B /win/salmon/user ${SIMG} \
    python src/train.py \
    --sign '' \
    --datalist 8 \
    --model 'VGG16' \
    --epoch 300 \
    --fold 4 \
    --image_size 224 \
    --batch_size 32 \
    --valid_batch_size 8 \
    --lr 1e-4 \
    --min_lr 5e-5 \
    --t_max 1350 \
    --t_0 25 \
    --wd 1e-4 \
    --num_classes 1 \
    --num_workers 2 \
    --num_sampling 1 \
    --optimizer 'Adam' \
    --criterion 'MAE Loss' \
    --seed 42 \
    --df_path '' \
    --datalist_path  '' \
    --image_path '' \
    --result_path ''"
# NORMAL DRR image or BONE DRR iamge
#--image_path '/win/salmon/user/masuda/project/vit_kl_crowe/20220511_DRR_with_Crowe_Kl/DRR_AP' \
#--image_path '/win/salmon/user/masuda/project/makedrr/150_Extracted_2D_DRR_944_masuda/DRR_AP' \

# t_max = epoch x batchsize
# 200 epoch t_max 1800
# 300 epoch t_max 2700
