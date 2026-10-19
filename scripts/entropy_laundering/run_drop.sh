#!/usr/bin/env bash
# Train full-length students with a fraction of response tokens dropped from the loss.
# Settings match /ephemeral/codex-textseal/length4096 (LoRA r32/a64, LR 5e-5, 1 epoch,
# effective batch 16, seed 42), plus --drop-rate. Usage: run_drop.sh GPUS PORT NAME:TRACES:RATE ...
set -euo pipefail
gpus=$1 port=$2; shift 2
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)  # cwd is this script's directory (cd above)
R=${FP_ROOT:-/ephemeral/codex-textseal}  # holds venv/, cache/huggingface, reproduction/hash.json
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$REPO \
  SCIENCE_TRAIN_PATH=$REPO/data/science/train_9k.jsonl \
  SCIENCE_EVAL_PATH=$REPO/data/science/eval_1k.jsonl \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpus TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpus \
  RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpus CUDA_VISIBLE_DEVICES=$gpus
mkdir -p logs students cache
for job in "$@"; do
  IFS=: read -r name traces rate <<< "$job"
  [ -e students/$name/adapter_model.safetensors ] && { echo "skip $name"; continue; }
  echo "$name train start $(date -u +%FT%TZ)" >> logs/stages.log
  $R/venv/bin/python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 \
    --master_port=$port --nproc_per_node=2 sft_tokendrop.py --drop-rate $rate \
    --dataset science --traces "$traces" --student-model Qwen/Qwen3.5-4B \
    --student-dtype bfloat16 --student-pad-token '' --gradient-checkpointing \
    --output-dir students/$name --epochs 1 --batch-size 1 --grad-accum 8 \
    --learning-rate 5e-05 --rank 32 --alpha 64 --dropout .05 --seed 42 \
    --max-seq-length 4096 --save-only-final ${EXTRA_ARGS:-} > logs/$name-train.log 2>&1
  echo "$name train done $(date -u +%FT%TZ)" >> logs/stages.log
done
