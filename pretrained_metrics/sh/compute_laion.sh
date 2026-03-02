#!/bin/bash
# compute_laion.sh
# Run from the project root: bash pretrained_metrics/sh/compute_laion.sh

BATCH_SIZE=${1:-16}

python pretrained_metrics/compute_pretrained_metrics.py \
    --dataset laion \
    --root "huggingface" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --output_dir "./results_anish/laion"
