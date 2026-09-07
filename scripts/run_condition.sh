#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STRATEGY_CONFIG="${1:?usage: run_condition.sh CONFIG [COMMON_CONFIG]}"
COMMON_CONFIG="${2:-${REPO_ROOT}/configs/science-qwen35.env}"
export REPO_ROOT
set -a
source "${COMMON_CONFIG}"
source "${STRATEGY_CONFIG}"
set +a

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

[[ -f "${TEACHER_ADAPTER}/adapter_model.safetensors" ]] || {
  echo "Missing trained teacher adapter. Run scripts/run_teacher_sft.sh first." >&2
  exit 2
}
comparison="${EXPERIMENT_DIR}/teacher_evals/sft/comparison_to_base.json"
"${CONDA_ENV_PREFIX}/bin/python" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); raise SystemExit(0 if d.get("passed") else 1)' \
  "${comparison}" || { echo "Teacher utility gate has not passed." >&2; exit 2; }

case "${METHOD}" in
  control) method_label=control ;;
  ads) method_label="ads-lambda${LAMBDA//./_}" ;;
  radioactive) method_label="radioactive-delta${DELTA//./_}" ;;
  *) echo "Unsupported METHOD=${METHOD}" >&2; exit 2 ;;
esac

cd "${REPO_ROOT}"
./pipeline.sh

lr_tag="$("${CONDA_ENV_PREFIX}/bin/python" -c 'import sys; print(f"{float(sys.argv[1]):g}")' "${LEARNING_RATE}")"
adapter="${EXPERIMENT_DIR}/models/${STUDENT_TAG_OVERRIDE}_${method_label}_lr${lr_tag}_e${EPOCHS}/student_lora"
base_eval="${EXPERIMENT_DIR}/utility_evals/base-student"
student_eval="${EXPERIMENT_DIR}/utility_evals/${method_label}"
mkdir -p "${base_eval}" "${student_eval}"

if [[ ! -f "${base_eval}/base.summary.json" ]]; then
  "${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run --standalone \
    --nproc_per_node="${ACC_NUM_PROCS}" "${REPO_ROOT}/scripts/eval_science_mcq.py" \
    --model "${STUDENT_MODEL}" --dataset "${SCIENCE_EVAL_PATH}" \
    --max-samples "${EVAL_SAMPLES}" --batch-size "${UTILITY_EVAL_BATCH}" \
    --max-new-tokens "${UTILITY_MAX_NEW_TOKENS}" --output "${base_eval}/base.jsonl"
fi
if [[ ! -f "${student_eval}/student.summary.json" ]]; then
  "${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run --standalone \
    --nproc_per_node="${ACC_NUM_PROCS}" "${REPO_ROOT}/scripts/eval_science_mcq.py" \
    --model "${STUDENT_MODEL}" --adapter "${adapter}" --dataset "${SCIENCE_EVAL_PATH}" \
    --max-samples "${EVAL_SAMPLES}" --batch-size "${UTILITY_EVAL_BATCH}" \
    --max-new-tokens "${UTILITY_MAX_NEW_TOKENS}" --output "${student_eval}/student.jsonl"
fi
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/compare_eval.py" \
  "${base_eval}/base.jsonl" "${student_eval}/student.jsonl" \
  --min-gain 0 --alpha 0.05 --output "${student_eval}/comparison_to_base.json"
