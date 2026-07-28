#!/usr/bin/env bash

set -euo pipefail
source /home/ningmang/miniconda3/etc/profile.d/conda.sh
conda activate eqvae_train

# get statis
CUDA_VISIBLE_DEVICES=4,5,6,7
torchrun --standalone --nproc_per_node=4 extract_REPA_latents.py \
  --task=stats \
  --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
  --config=configs/dsm_config_imgnet256.yaml \
  --checkpoint=exps/imagenet256_DSM_ft/checkpoints/40000.ckpt \
  --batch-size=128 \
  --num-stat-samples=200000


# # extract latents
# export CUDA_VISIBLE_DEVICES=4,5,6,7
# torchrun --standalone --nproc_per_node=4 extract_REPA_latents.py \
#   --task=extract \
#   --source=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256/train \
#   --dest=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_DSM_40k/vae-sd \
#   --dest-images=/mnt/vepfs/base2/stream/ningmang/datasets/imagenet256_DSM_40k/images \
#   --config=configs/dsm_config_imgnet256.yaml \
#   --checkpoint=exps/imagenet256_DSM_ft/checkpoints/40000.ckpt \
#   --batch-size=128


# sslaunch submit -j ningmang -n 1 -c zhongwei -q stream -- sleep infinity
# kubectl --kubeconfig /mnt/vepfs/base/infra/.kube/zhongweiconfig \
# -n default exec -it ningmang-ningmang-n1-260727095453-master-0 -- bash
# sudo chown -R ningmang:ningmang /home/ningmang/REPA

# bash extract_latents.sh