#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

OUTPUT_DIR="$SCRIPT_DIR/exps/imagenet256_DSMp03_ft_05lpips_10epochs"
LOG_DIR="$OUTPUT_DIR/logs/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/stdout.log") 2> >(tee -a "$LOG_DIR/stderr.log" >&2)
echo "stdout log: $LOG_DIR/stdout.log"
echo "stderr log: $LOG_DIR/stderr.log"

export TORCH_HOME=/home/ningmang/torch_cache
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export NCCL_DEBUG=WARN

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

torchrun --standalone --nproc_per_node=8 --master_port=29507 \
stage1_vae_trainer.py \
    --config configs/DSM_config_imgnet256.yaml \
    --output_dir "$OUTPUT_DIR" \
    --resume_path pretrained_models/model.ckpt \
    --batch_size 20 \
    --max_train_steps 80001 \
    --eval_interval_steps 2000 \
    --save_top_k 5


# nohup bash imagenet256_DSM.sh > /dev/null 2>&1 &
# pkill -TERM -f stage1_vae_trainer.py
# sslaunch-noproxy submit -j dsmp03 -c zhongwei -q stream -n 1 --no-log -- bash imagenet256_DSM.sh