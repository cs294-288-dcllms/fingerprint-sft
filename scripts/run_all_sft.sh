#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON_CONFIG="${1:-${REPO_ROOT}/configs/science-qwen35.env}"
export REPO_ROOT
set -a
source "${COMMON_CONFIG}"
set +a

"${REPO_ROOT}/scripts/run_teacher_sft.sh" "${COMMON_CONFIG}"
for strategy in control adfp-lambda8 adfp-lambda16 radioactive-delta2; do
  "${REPO_ROOT}/scripts/run_condition.sh" \
    "${REPO_ROOT}/configs/strategies/${strategy}.env" "${COMMON_CONFIG}"
done

lr_tag="$("${CONDA_ENV_PREFIX}/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${LEARNING_RATE}")"
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/stages/stage5_plotting.py" \
  --exp-dir "${EXPERIMENT_DIR}" --fig-dir "${EXPERIMENT_DIR}/figures" \
  --student-tag "${STUDENT_TAG_OVERRIDE}" --lr "${lr_tag}" --epochs "${EPOCHS}" \
  --show-labels --variants open_supervised open_unsupervised closed_supervised closed_unsupervised
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/summarize_sft.py" \
  --exp-dir "${EXPERIMENT_DIR}" --student-tag "${STUDENT_TAG_OVERRIDE}" \
  --lr "${LEARNING_RATE}" --epochs "${EPOCHS}"
