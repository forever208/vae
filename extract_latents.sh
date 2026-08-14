#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="$SCRIPT_DIR/exps/imagenet256_DSMp07_ft_05lpips_10epochs"
LOG_DIR="$OUTPUT_DIR/extract_logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

export TORCH_HOME=/home/ningmang/torch_cache
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN


# # extract latents
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
  --task=extract \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
  --dest=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_DSM_prob07_05lpips_72k/vae-sd \
  --dest-images=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_DSM_prob07_05lpips_72k/images \
  --config=configs/DSM_config_imgnet256.yaml \
  --checkpoint=/home/ningmang/vae/exps/imagenet256_DSMp07_ft_05lpips_10epochs/checkpoints/72000.ckpt \
  --batch-size=128


# get statis
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
torchrun --standalone --nproc_per_node=4 --master_port=29508 \
extract_REPA_latents.py \
  --task=stats \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
  --config=configs/DSM_config_imgnet256.yaml \
  --checkpoint=/home/ningmang/vae/exps/imagenet256_DSMp07_ft_05lpips_10epochs/checkpoints/72000.ckpt \
  --batch-size=128 \
  --num-stat-samples=200000

# sslaunch submit -j extract -c zhongwei -q stream -n 1 --no-log -- bash extract_latents.sh