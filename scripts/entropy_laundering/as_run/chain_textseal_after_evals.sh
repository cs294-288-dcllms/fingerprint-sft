#!/usr/bin/env bash
# GPUs 6,7 ran out of memory when training overlapped two evals and a detection. Wait until the
# current evals (textseal-drop90 on 6, synthid-drop0 on 7) and the synthid-drop90 detection are done,
# then train the TextSeal entropy-masked students, detect and evaluate them, and finally run the
# remaining evals (textseal-drop50, synthid-drop50, synthid-drop90).
set -uo pipefail
cd "$(dirname "$0")"
TS=/ephemeral/codex-textseal/reproduction/textseal/traces.jsonl
until [ -e results/textseal-drop90-eval-sampled.summary.json ] && [ -e results/synthid-drop0-eval-sampled.summary.json ] \
      && [ -e detection/synthid-drop90/summary.json ]; do sleep 120; done
for mode in top-entropy bottom-entropy; do
  EXTRA_ARGS="--mask-mode $mode --entropy-file entropy/textseal.npz" \
    ./run_drop.sh 6,7 29781 "textseal-$mode-50:$TS:0.5"
done
for mode in top-entropy bottom-entropy; do
  name=textseal-$mode-50
  NPROC=1 ./run_detect.sh 6 29786 "$name:students/$name:textseal:$TS" &
  ./run_eval.sh 7 "$name" &
  wait
done
./run_eval.sh 6 textseal-drop50 synthid-drop90 &
./run_eval.sh 7 synthid-drop50 &
wait
echo "chain textseal done $(date -u +%FT%TZ)" >> logs/stages.log
