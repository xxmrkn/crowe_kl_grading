#!/bin/bash

PROJECT=''
NUM_GPU=1
NUM_CORE=4
SIMG=''

mkdir -p "${PROJECT}/slurm_inf"

for i in 'VisionTransformer_Base16' 'VGG16' 'DenseNet161';
do 
NODE='cl-walrus'
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm_inf/slurm_${i}_${NODE}_%j.out \
  --wrap="singularity exec --nv -B /win/salmon/user  ${SIMG} \
    python src/inference.py \
    --sign '' \
    --datalist 8 \
    --model ${i} \
    --epoch 300 \
    --fold 4 \
    --image_size 224 \
    --batch_size 32 \
    --valid_batch_size 1 \
    --lr 5e-5 \
    --min_lr 5e-6 \
    --t_max 2700 \
    --t_0 25 \
    --wd 1e-4 \
    --num_classes 1 \
    --num_workers 4 \
    --num_sampling 1000 \
    --optimizer 'Adam' \
    --criterion 'MAE Loss' \
    --seed 42 \
    --df_path '' \
    --datalist_path  '' \
    --image_path '' \
    --result_path ''"
sleep 20s
done

# t_max = epoch x batchsize
# 200 epoch t_max 1800
# 300 epoch t_max 2700


# NODE='cl-yamaneko'
# sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
#   --wrap="singularity exec --nv -B /win/salmon/user ${SIMG} \
#     python umap.py --sign 0919_train --datalist 8"
