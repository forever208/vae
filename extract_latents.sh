#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

# get statis
torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
  --task=stats \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
  --config=configs/sdvae_config_imgnet256.yaml \
  --checkpoint=exps/imagenet256_SDVAE_ft/checkpoints/40000.ckpt \
  --batch-size=128 \
  --num-stat-samples=200000


# # extract latents
# torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
#   --task=extract \
#   --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
#   --dest=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_sdvae_40k/vae-sd \
#   --dest-images=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_sdvae_40k/images \
#   --config=configs/sdvae_config_imgnet256.yaml \
#   --checkpoint=exps/imagenet256_SDVAE_ft/checkpoints/40000.ckpt \
#   --batch-size=128


# bash extract_latents.sh