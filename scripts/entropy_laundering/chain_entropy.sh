#!/usr/bin/env bash
# Entropy-targeted masking: wait for the entropy file (and an optional training to finish),
# train two students at a 50% budget, then run detection and sampled eval for each.
# Usage: chain_entropy.sh GPU_A GPU_B PORT SCHEME TRACES ENTROPY_NPZ [WAIT_FOR_TRAIN_NAME]
set -uo pipefail
ga=$1 gb=$2 port=$3 scheme=$4 traces=$5 ent=$6 wait_for=${7:-}
cd "$(dirname "$0")"
until [ -e "$ent" ]; do sleep 120; done
[ -n "$wait_for" ] && until grep -q "^$wait_for train done" logs/stages.log; do sleep 120; done
names=()
for mode in top-entropy bottom-entropy; do
  name=$scheme-$mode-50
  names+=("$name")
  EXTRA_ARGS="--mask-mode $mode --entropy-file $ent" ./run_drop.sh $ga,$gb $port "$name:$traces:0.5"
done
for name in "${names[@]}"; do
  NPROC=1 ./run_detect.sh $ga $((port+5)) "$name:students/$name:$scheme:$traces" &
  ./run_eval.sh $gb "$name" &
  wait
done
echo "chain $scheme done $(date -u +%FT%TZ)" >> logs/stages.log
