#!/bin/bash

PROJECT='crowe_kl_classification'
NUM_GPU=1
NUM_CORE=4
SIMG='path'

mkdir -p "${PROJECT}/slurm"

NODE='node'
sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
  --wrap="singularity exec --nv -B /user ${SIMG} \
    python src/train.py --sign results --datalist 8 --model VisionTransformer_Base16 --epoch 1 --fold 4 \
    --image_size 224 --batch_size 32 --valid_batch_size 8 --lr 5e-5 --min_lr 3e-5 --t_max 1800 --t_0 25 \
    --wd 1e-4 --num_classes 7 --num_workers 4 --optimizer Adam --criterion 'Focal Loss' --seed 42 "


# NODE='cl-yamaneko'
# sbatch --gres=gpu:${NUM_GPU} -n ${NUM_CORE} -D ${PROJECT} -w ${NODE} -o slurm/slurm_${NODE}_%j.out \
#   --wrap="singularity exec --nv -B /win/salmon/user ${SIMG} \
#     python umap.py --sign 0919_train --datalist 8"