#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

cd /home/ningmang/eqvae/train_eqvae

OUTPUT_DIR=/home/ningmang/eqvae/exps/imagenet256_SDVAE_PyTorch
LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

export TORCH_HOME=/home/ningmang/torch_cache
export PYTHONPATH=/home/ningmang/eqvae/train_eqvae/src/taming-transformers:${PYTHONPATH:-}
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 --master_port=29507 \
train_pytorch.py --config configs/eqvae_config.yaml


# nohup bash imagenet256_EQVAE_pytorch_ddp.sh > /dev/null 2>&1 &