#!/bin/bash
# eda_vitonhd.sh
# Run from the project root: bash EDA/sh/eda_vitonhd.sh

ROOT=${1:-"./zalando-hd-resized"}
BATCH_SIZE=${2:-16}

python EDA/run_eda.py \
    --dataset vitonhd \
    --root "$ROOT" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --cache_dir "./eda_cache_anish" \
    --out_dir "./figures_anish/vitonhd"
