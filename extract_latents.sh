#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

# get statis
CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --standalone --nproc_per_node=4 extract_REPA_latents.py \
  --task=stats \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/celeba256 \
  --config=configs/SDVAE_config_celeba256.yaml \
  --checkpoint=exps/celeba256_SDVAE/checkpoints/98000.ckpt \
  --batch-size=128 \
  --num-stat-samples=200000


# # # extract latents
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# torchrun --standalone --nproc_per_node=8 extract_REPA_latents.py \
#   --task=extract \
#   --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
#   --dest=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_SDVAE/vae-sd \
#   --dest-images=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_SDVAE/images \
#   --config=configs/SDVAE_config_imgnet256.yaml \
#   --checkpoint=pretrained_models/model.ckpt \
#   --batch-size=128


# sslaunch submit -j ningmang -n 1 -c zhongwei -q stream -- sleep infinity
# kubectl --kubeconfig /mnt/vepfs/base/infra/.kube/zhongweiconfig \
# -n default exec -it ningmang-ningmang-n1-260801095136-master-0 -- bash
# sudo chown -R ningmang:ningmang /home/ningmang/REPA
# pkill -TERM -f extract_REPA_latents.py
# bash extract_latents.sh