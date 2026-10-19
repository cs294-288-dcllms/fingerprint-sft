#!/usr/bin/env bash
# ADFP detection (stage4: green-token probability, Hoeffding p-value) for one student.
# Usage: run_detect_adfp.sh GPU NAME ADAPTER TRACES MODE   (MODE = open | closed)
set -uo pipefail
gpu=$1 name=$2 adapter=$3 traces=$4 mode=$5
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)  # cwd is this script's directory (cd above)
R=${FP_ROOT:-/ephemeral/codex-textseal}  # holds venv/, cache/huggingface, reproduction/hash.json
out=detection-adfp/$name-$mode.json
[ -e "$out" ] && { echo "skip $name $mode"; exit 0; }
mkdir -p detection-adfp logs
CUDA_VISIBLE_DEVICES=$gpu HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
  PYTHONPATH=$REPO RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpu \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpu TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpu \
  $R/venv/bin/python $REPO/stages/stage4_watermark_eval.py --dataset science \
  --traces "$traces" --hash-config $R/reproduction/hash.json --teacher-model Qwen/Qwen3.5-9B \
  --teacher-pad-token '' --student-model Qwen/Qwen3.5-4B --student-pad-token '' --lora-dir "$adapter" \
  --mode $mode --supervision supervised --output "$out" > logs/detect-adfp-$name-$mode.log 2>&1
