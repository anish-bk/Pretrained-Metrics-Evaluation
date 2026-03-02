#!/bin/bash
# compute_dresscode.sh
# Run from the project root: bash pretrained_metrics/sh/compute_dresscode.sh

ROOT=${1:-"./dresscode"}
BATCH_SIZE=${2:-16}

python pretrained_metrics/compute_pretrained_metrics.py \
    --dataset dresscode \
    --root "$ROOT" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --output_dir "./results_anish/dresscode"
