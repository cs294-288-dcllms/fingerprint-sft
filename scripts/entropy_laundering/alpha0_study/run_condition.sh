#!/usr/bin/env bash
# Train a full-length (4096) student on one trace set, then run the sampled eval.
# Settings match /ephemeral/codex-textseal/length4096 except --save-only-final (disk).
# Usage: run_condition.sh NAME TRACES GPUS PORT
set -euo pipefail
name=$1 traces=$2 gpus=$3 port=$4
cd "$(dirname "$0")"
R=/ephemeral/codex-textseal
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=8 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  SCIENCE_TRAIN_PATH=$R/length-repo/data/science/train_9k.jsonl \
  SCIENCE_EVAL_PATH=$R/length-repo/data/science/eval_1k.jsonl \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpus TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpus \
  RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpus CUDA_VISIBLE_DEVICES=$gpus
mkdir -p logs cache
echo "train start $(date -u +%FT%TZ)" > logs/$name.stage
PYTHONPATH=$R/length-repo $R/venv/bin/python -m torch.distributed.run --nnodes=1 \
  --master_addr=127.0.0.1 --master_port=$port --nproc_per_node=2 \
  $R/length-repo/scripts/textseal_training_length.py --trainer-max-length 4096 \
  --dataset science --traces "$traces" --student-model Qwen/Qwen3.5-4B \
  --student-dtype bfloat16 --student-pad-token '' --gradient-checkpointing \
  --output-dir students/$name --epochs 1 --batch-size 1 --grad-accum 8 \
  --learning-rate 5e-05 --rank 32 --alpha 64 --dropout .05 --seed 42 \
  --max-seq-length 4096 --save-only-final > logs/$name-train.log 2>&1
echo "eval start $(date -u +%FT%TZ)" >> logs/$name.stage
PYTHONPATH=$R/repro-repo $R/venv/bin/python -m torch.distributed.run --nnodes=1 \
  --master_addr=127.0.0.1 --master_port=$((port+1)) --nproc_per_node=2 \
  ../eval_science_mcq_sampled.py --model Qwen/Qwen3.5-4B --adapter students/$name \
  --dataset $R/repro-repo/data/science/eval_1k.jsonl --max-samples 1000 \
  --batch-size 24 --max-new-tokens 2048 --output results/$name-eval-sampled.jsonl \
  > logs/$name-eval.log 2>&1
echo "done $(date -u +%FT%TZ)" >> logs/$name.stage
