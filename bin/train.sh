#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=4
#SIMG=''
SIMG=''

mkdir -p "${PROJECT}/slurm"

NODE='cl-dugong'

for i in 1;
do
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/datalist${i}_slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B /win/salmon/user -B /win/scallop/user ${SIMG} \
    python /project/20230616_crowe_kl_grading/src/train.py \
    --sign '0616_regression' \
    --base_path '' \
    --model 'VisionTransformer_Base16' \
    --pretrained False \
    --optimizer 'Adam' \
    --criterion 'MAE Loss' \
    --scheduler 'CosineAnnealingLR' \
    --datalist ${i} \
    --epoch 5 \
    --fold 4 \
    --image_size 224 \
    --batch_size 32 \
    --valid_batch_size 8 \
    --lr 8e-5 \
    --min_lr 1e-6 \
    --t_max 300 \
    --t_0 25 \
    --wd 1e-4 \
    --num_classes 1 \
    --num_workers 4 \
    --num_sampling 1 \
    --seed 42 \
    --df_path '' \
    --datalist_path  '' \
    --image_path '' \
    --result_path '' "
sleep 10s
done
