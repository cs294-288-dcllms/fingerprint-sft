#!/usr/bin/env bash
# Re-run the student science evals with Qwen3.5's recommended thinking-mode
# sampling (T=1.0, top_p=0.95, top_k=20, presence_penalty=1.5) instead of greedy.
# Control on GPUs 4,5 and TextSeal on GPUs 6,7 (GPUs 0-3 belong to another group); same 2,048-token cap as the greedy evals.
set -euo pipefail
cd "$(dirname "$0")"
R=/ephemeral/codex-textseal
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 RUNTIME_CACHE_ROOT=$PWD/runtime PYTHONPATH=$R/repro-repo
mkdir -p runtime results
run() {  # name gpus port
  CUDA_VISIBLE_DEVICES=$2 nohup $R/venv/bin/python -m torch.distributed.run --nnodes=1 \
    --master_addr=127.0.0.1 --master_port=$3 --nproc_per_node=2 eval_science_mcq_sampled.py \
    --model Qwen/Qwen3.5-4B --adapter $R/reproduction/$1-student \
    --dataset $R/repro-repo/data/science/eval_1k.jsonl --max-samples 1000 \
    --batch-size 24 --max-new-tokens 2048 \
    --output results/$1-eval-sampled.jsonl > results/$1-eval-sampled.log 2>&1 &
  echo "$1: GPUs $2, log results/$1-eval-sampled.log"
}
# Usage: ./run_sampled_evals.sh [control] [textseal]   (default: both)
for name in ${@:-control textseal}; do
  case $name in
    control) run control 4,5 29611 ;;
    textseal) run textseal 6,7 29612 ;;
    *) echo "unknown run: $name" >&2; exit 1 ;;
  esac
done
