#!/usr/bin/env bash
# Usage: run_flips.sh GPU NAME TRACES
set -euo pipefail
gpu=$1 name=$2 traces=$3
cd "$(dirname "$0")"
R=/ephemeral/codex-textseal
mkdir -p out logs cache
CUDA_VISIBLE_DEVICES=$gpu HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=$R/length-repo SCIENCE_TRAIN_PATH=$R/length-repo/data/science/train_9k.jsonl \
  SCIENCE_EVAL_PATH=$R/length-repo/data/science/eval_1k.jsonl TRITON_CACHE_DIR=$PWD/cache/triton-$gpu \
  TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpu \
  $R/venv/bin/python compute_flips.py "$traces" out/$name.npz --adapter $R/reproduction/teacher --max-rows 3000 \
  > logs/$name.log 2>&1
