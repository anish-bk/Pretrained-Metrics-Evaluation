#!/bin/bash
# eda_laion.sh
# Run from the project root: bash EDA/sh/eda_laion.sh

BATCH_SIZE=${1:-16}

python EDA/run_eda.py \
    --dataset laion \
    --root "huggingface" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --cache_dir "./eda_cache_anish" \
    --out_dir "./figures_anish/laion"
