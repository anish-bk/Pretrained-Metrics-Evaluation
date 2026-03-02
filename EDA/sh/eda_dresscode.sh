#!/bin/bash
# eda_dresscode.sh
# Run from the project root: bash EDA/sh/eda_dresscode.sh

ROOT=${1:-"./dresscode"}
BATCH_SIZE=${2:-16}

python EDA/run_eda.py \
    --dataset dresscode \
    --root "$ROOT" \
    --batch_size "$BATCH_SIZE" \
    --use_anish \
    --cache_dir "./eda_cache_anish" \
    --out_dir "./figures_anish/dresscode"
