#!/usr/bin/env bash
# Open-weight radioactivity on the first 3,000 training traces (teammate E1 protocol).
# Usage: run_detect.sh GPUS PORT JOB...   JOB = name:adapter:scheme:traces
set -euo pipefail
gpus=$1 port=$2; shift 2
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)  # cwd is this script's directory (cd above)
R=${FP_ROOT:-/ephemeral/codex-textseal}  # holds venv/, cache/huggingface, reproduction/hash.json
C=${DETECTOR_CODE:-/ephemeral/synthid-run/code}  # needs --scheme synthid support (synthid-text branch)
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  PYTHONPATH=$C CUDA_VISIBLE_DEVICES=$gpus RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpus \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpus TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpus
mkdir -p detection logs
for job in "$@"; do
  IFS=: read -r name adapter scheme traces <<< "$job"
  [ -e detection/$name/summary.json ] && { echo "skip $name"; continue; }
  # A detection started by hand elsewhere leaves RUNNING; don't start a second copy.
  [ -e detection/$name/RUNNING ] && { echo "skip $name (running elsewhere)"; continue; }
  $R/venv/bin/python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 \
    --master_port=$port --nproc_per_node=${NPROC:-2} $C/scripts/eval_textseal_radioactivity.py \
    --scheme $scheme --model Qwen/Qwen3.5-4B --adapter "$adapter" --trace-set known="$traces" \
    --output-dir detection/$name --max-samples 3000 --max-seq-length 4096 --batch-size 1 \
    --secret-key 294288 --secret-key-b 306633 --mixing-alpha 0.1 > logs/detect-$name.log 2>&1
done
