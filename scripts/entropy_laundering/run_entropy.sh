#!/usr/bin/env bash
# Usage: run_entropy.sh GPU TRACES OUT.npz
set -euo pipefail
gpu=$1 traces=$2 out=$3
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)  # cwd is this script's directory (cd above)
R=${FP_ROOT:-/ephemeral/codex-textseal}  # holds venv/, cache/huggingface, reproduction/hash.json
export CUDA_VISIBLE_DEVICES=$gpu HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 PYTHONPATH=$REPO \
  SCIENCE_TRAIN_PATH=$REPO/data/science/train_9k.jsonl SCIENCE_EVAL_PATH=$REPO/data/science/eval_1k.jsonl \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpu TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpu CUDA_CACHE_PATH=$PWD/cache/cuda-$gpu
$R/venv/bin/python compute_entropy.py "$traces" "$out"
