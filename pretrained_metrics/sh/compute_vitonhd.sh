#!/bin/bash
# compute_vitonhd.sh
# Run from the project root: bash pretrained_metrics/sh/compute_vitonhd.sh

ROOT=${1:-"./zalando-hd-resized"}
BATCH_SIZE=${2:-16}

python pretrained_metrics/compute_pretrained_metrics.py \
    --dataset vitonhd \
    --root "$ROOT" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --output_dir "./results_anish/vitonhd"
