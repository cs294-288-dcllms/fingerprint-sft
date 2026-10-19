#!/usr/bin/env bash
# Sampled MCQ eval (Qwen3.5 thinking-mode settings, 2,048-token cap) for token-drop students,
# one GPU per queue. Waits for each student's training to finish. Usage: run_eval.sh GPU NAME...
set -uo pipefail
gpu=$1; shift
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)  # cwd is this script's directory (cd above)
R=${FP_ROOT:-/ephemeral/codex-textseal}  # holds venv/, cache/huggingface, reproduction/hash.json
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=$REPO CUDA_VISIBLE_DEVICES=$gpu RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpu
mkdir -p results logs
for name in "$@"; do
  [ -e results/$name-eval-sampled.summary.json ] && { echo "skip $name"; continue; }
  # Another queue already started this eval; don't run a second copy.
  [ -e results/$name.RUNNING ] && { echo "skip $name (running elsewhere)"; continue; }
  until grep -q "^$name train done" logs/stages.log; do sleep 120; done
  touch results/$name.RUNNING
  echo "$name eval start $(date -u +%FT%TZ)" >> logs/stages.log
  $R/venv/bin/python eval_science_mcq_sampled.py --model Qwen/Qwen3.5-4B \
    --adapter students/$name --dataset $REPO/data/science/eval_1k.jsonl \
    --max-samples 1000 --batch-size ${EVAL_BATCH:-24} --max-new-tokens 2048 \
    --output results/$name-eval-sampled.jsonl > logs/$name-eval.log 2>&1
  rm -f results/$name.RUNNING
  echo "$name eval done $(date -u +%FT%TZ)" >> logs/stages.log
done
