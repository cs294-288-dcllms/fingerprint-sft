#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCIENCE_CONFIG="${1:-${REPO_ROOT}/configs/science-qwen35.env}"
ROBUSTNESS_CONFIG="${TEACHER_SEED_ROBUSTNESS_CONFIG:-${REPO_ROOT}/configs/eval/teacher-seed-robustness.env}"

source "${SCIENCE_CONFIG}"
source "${ROBUSTNESS_CONFIG}"
EXPERIMENT_DIR="${TEACHER_SEED_EXPERIMENT_DIR:-${EXPERIMENT_DIR}}"

CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-${REPO_ROOT}/outputs/conda/fingerprint-sft}"
PYTHON="${CONDA_ENV_PREFIX}/bin/python"
ACCELERATE="${CONDA_ENV_PREFIX}/bin/accelerate"
RESULTS_ROOT="${TEACHER_SEED_RESULTS_ROOT:-${EXPERIMENT_DIR}/teacher_seed_robustness}"
TRAIN_TRACES="${EXPERIMENT_DIR}/training_traces/ads-lambda16/traces.jsonl"
LEGACY_SEED43_TRACES="${EXPERIMENT_DIR}/alternative_traces/ads-lambda16/traces.jsonl"
LENGTH_HINTS="${TEACHER_RESAMPLE_LENGTH_HINTS:-${EXPERIMENT_DIR}/alternative_traces/control/traces.jsonl}"
HASH_CONFIG="${EXPERIMENT_DIR}/hash_seed/hash_config.json"
SFT_ADAPTER="${TEACHER_SEED_SFT_ADAPTER:-${EXPERIMENT_DIR}/models/${STUDENT_TAG_OVERRIDE}_ads-lambda16_lr5e-05_e1/student_lora}"
OPD_CONTROL_ROOT="${TEACHER_SEED_OPD_CONTROL_ROOT:-${EXPERIMENT_DIR}/opd/control-teacher}"
OPD_ADFP_ROOT="${TEACHER_SEED_OPD_ADFP_ROOT:-${EXPERIMENT_DIR}/opd/adfp-teacher-lambda16}"

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SCIENCE_TRAIN_PATH SCIENCE_EVAL_PATH
export HF_HOME="${HF_HOME:-${REPO_ROOT}/outputs/model-cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${REPO_ROOT}/outputs/xdg-cache}"
export RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${REPO_ROOT}/outputs/runtime-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"
export TMPDIR="${TMPDIR:-/tmp}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
mkdir -p \
  "${TRITON_CACHE_DIR}" \
  "${TORCHINDUCTOR_CACHE_DIR}" \
  "${CUDA_CACHE_PATH}"

for required in \
  "${PYTHON}" \
  "${ACCELERATE}" \
  "${TRAIN_TRACES}" \
  "${LENGTH_HINTS}" \
  "${HASH_CONFIG}" \
  "${TEACHER_ADAPTER}" \
  "${SFT_ADAPTER}/adapter_config.json"; do
  [[ -e "${required}" ]] || { echo "Missing teacher-seed input: ${required}" >&2; exit 2; }
done

declare -a target_specs=(
  "sft-lambda16|λ16 SFT|${SFT_ADAPTER}"
  "opd-control-step${TEACHER_RESAMPLE_OPD_STEP}|OPD control teacher, step ${TEACHER_RESAMPLE_OPD_STEP}|${OPD_CONTROL_ROOT}/exports/global_step_${TEACHER_RESAMPLE_OPD_STEP}/policy"
  "opd-adfp-step${TEACHER_RESAMPLE_OPD_STEP}|OPD ADFP teacher, step ${TEACHER_RESAMPLE_OPD_STEP}|${OPD_ADFP_ROOT}/exports/global_step_${TEACHER_RESAMPLE_OPD_STEP}/policy"
)

mkdir -p "${RESULTS_ROOT}/pairs" "${RESULTS_ROOT}/targets"
read -r -a seeds <<< "${TEACHER_RESAMPLE_SEEDS}"

for seed in "${seeds[@]}"; do
  if [[ "${seed}" == "43" && -f "${LEGACY_SEED43_TRACES}" ]]; then
    traces="${LEGACY_SEED43_TRACES}"
  else
    trace_dir="${RESULTS_ROOT}/traces/seed${seed}"
    traces="${trace_dir}/traces.jsonl"
    metadata="${trace_dir}/metadata.json"
    mkdir -p "${trace_dir}"
    if [[ ! -s "${traces}" || ! -s "${metadata}" ]]; then
      "${ACCELERATE}" launch \
        --config_file "${REPO_ROOT}/accelerate_config.yaml" \
        --num_processes "${ACC_NUM_PROCS}" \
        "${REPO_ROOT}/stages/stage1_generate.py" \
        --dataset science \
        --split train \
        --max-examples "${TEACHER_RESAMPLE_EXAMPLES}" \
        --teacher-model "${TEACHER_MODEL}" \
        --teacher-adapter "${TEACHER_ADAPTER}" \
        --teacher-dtype "${TEACHER_DTYPE}" \
        --teacher-pad-token "" \
        --proxy-model "${PROXY_MODEL}" \
        --proxy-dtype "${PROXY_DTYPE}" \
        --proxy-pad-token "" \
        --method ads \
        --lam 16 \
        --hash-config "${HASH_CONFIG}" \
        --output "${traces}" \
        --metadata "${metadata}" \
        --batch-size "${TEACHER_RESAMPLE_BATCH_SIZE}" \
        --seed "${seed}" \
        --max-new-tokens "${MAX_NEW_TOKENS}" \
        --temperature 0.7 \
        --top-p 0.95 \
        --repetition-penalty 1.0 \
        --length-hints "${LENGTH_HINTS}"
    fi
  fi

  pair_report="${RESULTS_ROOT}/pairs/seed${seed}.json"
  "${PYTHON}" "${REPO_ROOT}/scripts/check_paired_traces.py" \
    --right-is-prefix \
    --expected-left-seed 42 \
    --expected-right-seed "${seed}" \
    --output "${pair_report}" \
    "${TRAIN_TRACES}" "${traces}"

  for target_spec in "${target_specs[@]}"; do
    IFS='|' read -r target_name target_label adapter <<< "${target_spec}"
    [[ -f "${adapter}/adapter_config.json" ]] || {
      echo "Missing target adapter: ${adapter}" >&2
      exit 2
    }
    target_root="${RESULTS_ROOT}/targets/${target_name}"
    mkdir -p "${target_root}/seed${seed}"
    "${PYTHON}" - "${target_root}/target.json" "${target_label}" "${adapter}" <<'PY'
import json
import sys
from pathlib import Path

path, label, adapter = sys.argv[1:]
Path(path).write_text(
    json.dumps({"label": label, "adapter": str(Path(adapter).resolve())}, indent=2)
    + "\n",
    encoding="utf-8",
)
PY
    student_model="$("${PYTHON}" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' \
      "${adapter}/adapter_config.json")"

    for mode in open closed; do
      output="${target_root}/seed${seed}/watermark_${mode}.json"
      [[ -s "${output}" ]] && continue
      if [[ "${seed}" == "43" ]]; then
        if [[ "${target_name}" == "sft-lambda16" ]]; then
          existing="${EXPERIMENT_DIR}/metrics/${STUDENT_TAG_OVERRIDE}_ads-lambda16_lr5e-05_e1/watermark_${mode}_unsupervised.json"
        elif [[ "${target_name}" == opd-control-* ]]; then
          existing="${OPD_CONTROL_ROOT}/fingerprint_evals/global_step_${TEACHER_RESAMPLE_OPD_STEP}/watermark_${mode}_unsupervised.json"
        else
          existing="${OPD_ADFP_ROOT}/fingerprint_evals/global_step_${TEACHER_RESAMPLE_OPD_STEP}/watermark_${mode}_unsupervised.json"
        fi
        if [[ -s "${existing}" ]]; then
          cp -p "${existing}" "${output}"
          continue
        fi
      fi
      "${ACCELERATE}" launch \
        --config_file "${REPO_ROOT}/accelerate_config.yaml" \
        --num_processes "${ACC_NUM_PROCS}" \
        "${REPO_ROOT}/stages/stage4_watermark_eval.py" \
        --traces "${traces}" \
        --hash-config "${HASH_CONFIG}" \
        --teacher-model "${TEACHER_MODEL}" \
        --teacher-dtype "${TEACHER_DTYPE}" \
        --teacher-pad-token "" \
        --student-model "${student_model}" \
        --student-dtype "${STUDENT_DTYPE}" \
        --student-pad-token "" \
        --lora-dir "${adapter}" \
        --mode "${mode}" \
        --supervision unsupervised \
        --output "${output}" \
        --batch-size "${TEACHER_RESAMPLE_EVAL_BATCH_SIZE}" \
        --seed "${seed}" \
        --dataset science
    done
  done
done

"${PYTHON}" "${REPO_ROOT}/scripts/summarize_teacher_seed_robustness.py" \
  --results-root "${RESULTS_ROOT}" \
  --seeds "${seeds[@]}" \
  --output "${RESULTS_ROOT}/verified_complete.json" \
  --markdown "${RESULTS_ROOT}/results.md"
