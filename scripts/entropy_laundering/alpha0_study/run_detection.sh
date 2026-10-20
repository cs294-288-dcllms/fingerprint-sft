#!/usr/bin/env bash
# Known-trace (open-weight, teacher-forcing) and free-generation detection for the alpha0 study.
# Known traces: the 419 follow-up "known" prompts that survive the alpha0 repetition filter.
# alpha0 = key B only (models/textseal.py routes to key A with probability alpha).
# Usage: run_detection.sh GPUS PORT  -- runs every whitebox job sequentially on those GPUs.
set -euo pipefail
gpus=$1 port=$2
cd "$(dirname "$0")"
R=/ephemeral/codex-textseal
export HF_HOME=$R/cache/huggingface HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=8 \
  PYTHONPATH=$R/followup-repo CUDA_VISIBLE_DEVICES=$gpus RUNTIME_CACHE_ROOT=$PWD/cache/runtime-$gpus \
  TRITON_CACHE_DIR=$PWD/cache/triton-$gpus TORCHINDUCTOR_CACHE_DIR=$PWD/cache/inductor-$gpus
whitebox() {  # name adapter traces alpha
  [ -e detection/$1/summary.json ] && return
  $R/venv/bin/python -m torch.distributed.run --nnodes=1 --master_addr=127.0.0.1 \
    --master_port=$port --nproc_per_node=2 $R/followup-repo/scripts/eval_textseal_radioactivity.py \
    --model Qwen/Qwen3.5-4B --adapter $2 --trace-set known=$3 --output-dir detection/$1 \
    --max-seq-length 4096 --batch-size 1 --mixing-alpha $4 \
    --secret-key 294288 --secret-key-b 306633 > logs/detect-$1.log 2>&1
}
mkdir -p detection logs
shift 2
for job in "$@"; do
  case $job in
    a0u) whitebox alpha0-unfiltered students/alpha0-unfiltered data/known-alpha0.jsonl 0 ;;
    a0f) whitebox alpha0-filtered students/alpha0-filtered data/known-alpha0.jsonl 0 ;;
    a01) whitebox alpha01-length4096 $R/length4096/textseal-student data/known-alpha01.jsonl 0.1 ;;
    ctl0) whitebox control-vs-alpha0 $R/length4096/control-student data/known-alpha0.jsonl 0 ;;
    ctl01) whitebox control-vs-alpha01 $R/length4096/control-student data/known-alpha01.jsonl 0.1 ;;
    # 1,024-trained students from reproduction/ (readable); null reference / old alpha0.1 baseline.
    rctl0) whitebox control1024-vs-alpha0 $R/reproduction/control-student data/known-alpha0.jsonl 0 ;;
    r01) whitebox alpha01-length1024 $R/reproduction/textseal-student data/known-alpha01.jsonl 0.1 ;;
  esac
done
