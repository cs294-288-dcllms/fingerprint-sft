#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON_CONFIG="${1:-${REPO_ROOT}/configs/science-qwen35.env}"
TEACHER_CONFIG="${2:-${REPO_ROOT}/configs/teacher-sft.env}"
export REPO_ROOT
set -a
source "${COMMON_CONFIG}"
source "${TEACHER_CONFIG}"
set +a

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "${EXPERIMENT_DIR}/teacher_evals/base" "${EXPERIMENT_DIR}/teacher_evals/sft" "${TEACHER_ADAPTER}"
STATUS_FILE="${EXPERIMENT_DIR}/teacher/status.env"
mkdir -p "$(dirname "${STATUS_FILE}")"

write_status() {
  local state="$1" detail="$2" temporary="${STATUS_FILE}.tmp"
  {
    printf 'state=%s\n' "${state}"
    printf 'detail=%s\n' "${detail}"
    printf 'updated_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    printf 'teacher_adapter=%s\n' "${TEACHER_ADAPTER}"
  } > "${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}
trap 'rc=$?; write_status failed "line=${LINENO} exit=${rc}"; exit ${rc}' ERR

for required in "${SCIENCE_EVAL_PATH}" "${GOLD_SFT_PATH}"; do
  [[ -f "${required}" ]] || { echo "Missing ${required}; run scripts/prepare_science.sh" >&2; exit 2; }
done

BASE_DIR="${EXPERIMENT_DIR}/teacher_evals/base"
SFT_DIR="${EXPERIMENT_DIR}/teacher_evals/sft"
if [[ ! -f "${BASE_DIR}/teacher.summary.json" ]]; then
  write_status evaluating-base "1000-question base-teacher evaluation"
  "${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run --standalone \
    --nproc_per_node="${ACC_NUM_PROCS}" "${REPO_ROOT}/scripts/eval_science_mcq.py" \
    --model "${TEACHER_MODEL}" --dataset "${SCIENCE_EVAL_PATH}" \
    --max-samples "${EVAL_SAMPLES}" --batch-size "${TEACHER_EVAL_BATCH}" \
    --max-new-tokens "${TEACHER_EVAL_MAX_NEW_TOKENS}" \
    --output "${BASE_DIR}/teacher.jsonl"
fi

if [[ ! -f "${TEACHER_ADAPTER}/adapter_model.safetensors" ]]; then
  write_status training "teacher SFT on gold science reasoning traces"
  checkpoint_args=()
  [[ "${TEACHER_GRADIENT_CHECKPOINTING}" == 1 ]] && checkpoint_args=(--gradient-checkpointing)
  "${CONDA_ENV_PREFIX}/bin/accelerate" launch \
    --config_file "${REPO_ROOT}/accelerate_config.yaml" \
    --num_processes "${ACC_NUM_PROCS}" "${REPO_ROOT}/stages/stage3_finetune.py" \
    --dataset science --traces "${GOLD_SFT_PATH}" \
    --student-model "${TEACHER_MODEL}" --student-dtype "${TEACHER_DTYPE}" \
    --student-pad-token "" "${checkpoint_args[@]}" \
    --output-dir "${TEACHER_ADAPTER}" --epochs "${TEACHER_SFT_EPOCHS}" \
    --batch-size "${TEACHER_SFT_BATCH}" --grad-accum "${TEACHER_SFT_GRAD_ACCUM}" \
    --learning-rate "${TEACHER_SFT_LR}" --rank "${TEACHER_SFT_RANK}" \
    --alpha "${TEACHER_SFT_ALPHA}" --dropout "${TEACHER_SFT_DROPOUT}" \
    --seed "${TRAIN_SEED}" --max-seq-length "${TEACHER_SFT_MAX_SEQ_LEN}"
fi

if [[ ! -f "${SFT_DIR}/teacher.summary.json" ]]; then
  write_status evaluating-sft "1000-question trained-teacher evaluation"
  "${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run --standalone \
    --nproc_per_node="${ACC_NUM_PROCS}" "${REPO_ROOT}/scripts/eval_science_mcq.py" \
    --model "${TEACHER_MODEL}" --adapter "${TEACHER_ADAPTER}" \
    --dataset "${SCIENCE_EVAL_PATH}" --max-samples "${EVAL_SAMPLES}" \
    --batch-size "${TEACHER_EVAL_BATCH}" --max-new-tokens "${TEACHER_EVAL_MAX_NEW_TOKENS}" \
    --output "${SFT_DIR}/teacher.jsonl"
fi

write_status comparing "paired exact McNemar teacher gate"
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/compare_eval.py" \
  "${BASE_DIR}/teacher.jsonl" "${SFT_DIR}/teacher.jsonl" \
  --min-gain 0 --alpha 0.05 --output "${SFT_DIR}/comparison_to_base.json"
write_status complete "trained teacher significantly improves held-out utility"
