#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train


export TORCH_HOME=/home/ningmang/torch_cache
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN


CUDA_VISIBLE_DEVICES=0 python vavae_visualize_rFID.py \
  --config configs/VAVAE_config_imgnet256.yaml \
  --checkpoint exps/imagenet256_VAVAE/vavae-imagenet256-f16d32-dinov2.pt \
  --data-root /mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/val \
  --eval-batch-size 32 \
  --output vavae_reconstructions.png

# bash vavae_visualize_rFID.sh