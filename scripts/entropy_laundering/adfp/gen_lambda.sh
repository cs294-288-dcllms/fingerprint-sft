#!/usr/bin/env bash
# ADFP (ADS) teacher traces for one lambda on the 9,000 science prompts (same settings as the
# TextSeal/SynthID reproduction). Usage: gen_lambda.sh LAMBDA GPUS NPROC PORT
set -uo pipefail
lam=$1 gpus=$2 nproc=$3 port=$4
cd "$(dirname "$0")"
R=/ephemeral/codex-textseal
out=traces/lambda$lam
mkdir -p $out logs
echo "lambda$lam generate start $(date -u +%FT%TZ)" >> logs/stages.log
HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$R/length-repo CUDA_VISIBLE_DEVICES=$gpus \
SCIENCE_TRAIN_PATH=$R/length-repo/data/science/train_9k.jsonl SCIENCE_EVAL_PATH=$R/length-repo/data/science/eval_1k.jsonl \
TRITON_CACHE_DIR=$PWD/cache/triton TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor RUNTIME_CACHE_ROOT=$PWD/cache/runtime \
$R/venv/bin/python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 --master_port=$port \
  --nproc_per_node=$nproc $R/length-repo/stages/stage1_generate.py --dataset science --split train \
  --max-examples 9000 --teacher-model Qwen/Qwen3.5-9B --teacher-adapter $R/reproduction/teacher \
  --teacher-dtype bfloat16 --proxy-model Qwen/Qwen3.5-4B --teacher-pad-token '' --proxy-pad-token '' \
  --method ads --lam $lam --hash-config $R/reproduction/hash.json \
  --output $out/traces.jsonl --metadata $out/metadata.json --batch-size 24 --seed 42 \
  --max-new-tokens 3840 --temperature 0.7 --top-p 0.95 --repetition-penalty 1.0 --length-hints off \
  > logs/generate-lambda$lam.log 2>&1
echo "lambda$lam generate done (exit $?) $(date -u +%FT%TZ)" >> logs/stages.log
