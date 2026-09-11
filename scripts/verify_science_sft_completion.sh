#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON_CONFIG="${1:-${REPO_ROOT}/configs/science-qwen35.env}"
export REPO_ROOT

set -a
source "${COMMON_CONFIG}"
set +a

SUMMARY="${SFT_SUMMARY:-${EXPERIMENT_DIR}/summary.json}"
PAIR_REPORT="${SFT_PAIR_REPORT:-${EXPERIMENT_DIR}/metrics/same_prompt_teacher_seed_pair.json}"
MANIFEST="${SFT_MANIFEST:-${EXPERIMENT_DIR}/sft_verified_complete.json}"

"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/verify_sft_summary.py" \
  --summary "${SUMMARY}" \
  --pair-report "${PAIR_REPORT}" \
  --manifest "${MANIFEST}" \
  --train-seed "${TRAIN_SEED}" \
  --alt-seed "${ALT_SEED}" \
  --train-examples "${NUM_EXAMPLES}" \
  --alt-examples "${ALT_NUM_EXAMPLES}" \
  --eval-samples "${EVAL_SAMPLES}"
