#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train


export TORCH_HOME=/home/ningmang/torch_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN

CONFIG=configs/SDVAE_config_imgnet256.yaml \
CHECKPOINT=pretrained_models/model.ckpt \

CUDA_VISIBLE_DEVICES=0 python visualize_and_rFID.py \
  --config "${CONFIG}" \
  --checkpoint "${CHECKPOINT}" \
  --data-root /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/val \
  --eval-batch-size 32 \
  --output reconstructions.png

# VA-VAE:
# CONFIG="${CONFIG:-configs/VAVAE_config_imgnet256.yaml}"
# CHECKPOINT="${CHECKPOINT:-exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt}"


# SDVAE:
# CONFIG=configs/SDVAE_config_imgnet256.yaml \
# CHECKPOINT=pretrained_models/model.ckpt \


# EQ-VAE:
# CONFIG=configs/EQVAE_config_imgnet256.yaml \
# CHECKPOINT=exps/imagenet256_EQVAE_ft/checkpoints/40000.ckpt \