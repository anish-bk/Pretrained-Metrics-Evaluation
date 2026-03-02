#!/bin/bash
# eda_viton.sh
# Run from the project root: bash EDA/sh/eda_viton.sh

ROOT=${1:-"./VITON"}
BATCH_SIZE=${2:-16}

python EDA/run_eda.py \
    --dataset viton \
    --root "$ROOT" \
    --batch_size "$BATCH_SIZE" \
    --cache_dir "./eda_cache" \
    --out_dir "./figures/viton"
