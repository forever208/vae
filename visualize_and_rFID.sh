#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train


export TORCH_HOME=/home/ningmang/torch_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN

CONFIG=configs/DSM_config_imgnet256.yaml
CHECKPOINT=exps/imagenet256_DSM_ft_05lpips/checkpoints/36000.ckpt

torchrun --standalone --nproc_per_node=8 visualize_and_rFID.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --data-root /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/val \
  --lpfid-masks 0 2 4 6 8 \
  --eval-batch-size 32 \
  --output reconstructions.png

# Eight-GPU evaluation:
# NUM_GPUS=8 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
# bash visualize_and_rFID.sh

# SDVAE:
# CONFIG=configs/SDVAE_config_imgnet256.yaml
# CHECKPOINT=exps/imagenet256_SDVAE_ft/checkpoints/40000.ckpt

# EQ-VAE:
# CONFIG=configs/EQVAE_config_imgnet256.yaml
# CHECKPOINT=exps/imagenet256_EQVAE_ft/checkpoints/40000.ckpt

# VA-VAE:
# CONFIG="${CONFIG:-configs/VAVAE_config_imgnet256.yaml}"
# CHECKPOINT="${CHECKPOINT:-exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt}"

# DSM:
# CONFIG=configs/DSM_config_imgnet256.yaml
# CHECKPOINT=exps/imagenet256_DSM_ft_05lpips/checkpoints/36000.ckpt