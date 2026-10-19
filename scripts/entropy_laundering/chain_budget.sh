#!/usr/bin/env bash
# High-entropy masking at a given budget: train, then detection (one GPU) and sampled eval (other GPU).
# Usage: chain_budget.sh GPU_A GPU_B PORT SCHEME TRACES ENTROPY_NPZ RATE TAG
set -uo pipefail
ga=$1 gb=$2 port=$3 scheme=$4 traces=$5 ent=$6 rate=$7 tag=$8
cd "$(dirname "$0")"
name=$scheme-top-entropy-$tag
EXTRA_ARGS="--mask-mode top-entropy --entropy-file $ent ${CHAIN_EXTRA:-}" ./run_drop.sh $ga,$gb $port "$name:$traces:$rate"
NPROC=1 ./run_detect.sh $ga $((port+5)) "$name:students/$name:$scheme:$traces" &
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BATCH=12 ./run_eval.sh $gb "$name" &
wait
echo "chain $name done $(date -u +%FT%TZ)" >> logs/stages.log
