#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export REPO_ROOT
set -a
source "${REPO_ROOT}/configs/science-qwen35.env"
set +a

mkdir -p "${REPO_ROOT}/data/local"
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/prepare_science_thinking.py" \
  --train-output "${SCIENCE_TRAIN_PATH}" \
  --eval-output "${SCIENCE_EVAL_PATH}" \
  --train-samples "${NUM_EXAMPLES}" \
  --eval-samples "${EVAL_SAMPLES}" \
  --max-tokens "${MAX_SEQ_LEN}" \
  --seed "${TRAIN_SEED}"
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/prepare_science_gold_sft.py" \
  --input "${SCIENCE_TRAIN_PATH}" \
  --output "${GOLD_SFT_PATH}" \
  --tokenizer "${STUDENT_MODEL}"
