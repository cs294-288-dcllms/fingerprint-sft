#!/usr/bin/env bash
# ADFP lambda-64: unmasked student (already training on GPUs 4,5) and highest-entropy 50% (GPUs 6,7);
# then ADFP detection (first 3,000 traces) and sampled eval for each on GPUs 0-3.
set -uo pipefail
cd "$(dirname "$0")"
A=$HOME/textseal-evals/adfp
TR=$A/traces/lambda64/traces.jsonl
DT=$A/traces/lambda64/detect-3000.jsonl
ENT=entropy/adfp64.npz
post() {  # post NAME DETECT_GPU EVAL_GPU (after training is logged as done)
  until grep -q "^$1 train done" logs/stages.log; do sleep 60; done
  ./run_detect_adfp.sh $2 $1 students/$1 $DT open
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EVAL_BATCH=12 ./run_eval.sh $3 $1
}
post adfp64-drop0 0 1 &
( until grep -q "^adfp64 entropy done" logs/stages.log; do sleep 60; done
  EXTRA_ARGS="--entropy-file $ENT --mask-mode top-entropy" ./run_drop.sh 6,7 29961 "adfp64-top-entropy-50:$TR:0.5"
  post adfp64-top-entropy-50 3 2 ) &
wait
echo "adfp64 main done $(date -u +%FT%TZ)" >> logs/stages.log
