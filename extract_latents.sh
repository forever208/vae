#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="$SCRIPT_DIR/exps/celeba256_SDVAE"
LOG_DIR="$OUTPUT_DIR/extract_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

export TORCH_HOME=/home/ningmang/torch_cache
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"


torchrun --standalone --nproc_per_node=8 extract_latents.py \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/celeba256/celeba256 \
  --dest=/mnt/vepfs/base2/stream/ningmang/datasets/celeba256_latents/SDVAE_98k \
  --config=configs/SDVAE_config_celeba256.yaml \
  --checkpoint=exps/celeba256_SDVAE/checkpoints/98000.ckpt \
  --batch-size=100 \
  --num-workers=4 \
  --num-stat-samples=30000


# sslaunch submit -j extract -c zhongwei -q stream -n 1 --no-log -- bash extract_latents.sh