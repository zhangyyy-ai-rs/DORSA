#!/usr/bin/env bash

set -euo pipefail

DATA_PATH=${1:-/path/to/imagenet}
OUTPUT_DIR=${2:-./outputs/imagenet_dorsa}
GPUS=${3:-8}

torchrun --nproc_per_node="${GPUS}" pretrain/main.py \
  --data-path "${DATA_PATH}" \
  --model DORSA_T_2262_s48 \
  --batch-size 256 \
  --epochs 300 \
  --lr 1e-3 \
  --weight-decay 0.05 \
  --amp \
  --sync-bn \
  --output-dir "${OUTPUT_DIR}"
