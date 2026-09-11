#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPD_ROOT="${1:?usage: eval_science_opd_fingerprints.sh OPD_ROOT}"
SFT_CONDA_ENV_PREFIX="${SFT_CONDA_ENV_PREFIX:-${REPO_ROOT}/outputs/conda/fingerprint-sft}"
EXPERIMENT_DIR="${OPD_EXPERIMENT_DIR:-$(cd "${OPD_ROOT}/../.." && pwd)}"
TRAIN_TRACES="${OPD_SUPERVISED_TRACES:-${EXPERIMENT_DIR}/training_traces/ads-lambda16/traces.jsonl}"
ALT_TRACES="${OPD_UNSUPERVISED_TRACES:-${EXPERIMENT_DIR}/alternative_traces/ads-lambda16/traces.jsonl}"
HASH_CONFIG="${OPD_HASH_CONFIG:-${EXPERIMENT_DIR}/hash_seed/hash_config.json}"
TEACHER_MODEL="${TEACHER_MODEL:-Qwen/Qwen3.5-9B}"
NUM_GPUS="${OPD_NUM_GPUS:-8}"
BATCH_SIZE="${OPD_FINGERPRINT_BATCH_SIZE:-12}"
METRICS_ROOT="${OPD_ROOT}/fingerprint_evals"
RUNTIME_CACHE_ROOT="${RUNTIME_CACHE_ROOT:-${OPD_ROOT}/runtime-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${RUNTIME_CACHE_ROOT}/torchinductor}"
export CUDA_CACHE_PATH="${CUDA_CACHE_PATH:-${RUNTIME_CACHE_ROOT}/cuda}"
mkdir -p "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}" "${CUDA_CACHE_PATH}"

for required in \
  "${SFT_CONDA_ENV_PREFIX}/bin/python" \
  "${SFT_CONDA_ENV_PREFIX}/bin/accelerate" \
  "${TRAIN_TRACES}" \
  "${ALT_TRACES}" \
  "${HASH_CONFIG}"; do
  [[ -e "${required}" ]] || { echo "Missing fingerprint-eval input: ${required}" >&2; exit 2; }
done

mkdir -p "${METRICS_ROOT}"
PAIR_REPORT="${METRICS_ROOT}/same_prompt_teacher_seed_pair.json"
"${SFT_CONDA_ENV_PREFIX}/bin/python" \
  "${REPO_ROOT}/scripts/check_paired_traces.py" \
  --right-is-prefix \
  --expected-left-seed 42 \
  --expected-right-seed 43 \
  --output "${PAIR_REPORT}" \
  "${TRAIN_TRACES}" "${ALT_TRACES}"
while IFS= read -r policy; do
  export_step="$(basename "$(dirname "${policy}")")"
  metrics_dir="${METRICS_ROOT}/${export_step}"
  student_model="$("${SFT_CONDA_ENV_PREFIX}/bin/python" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))["base_model_name_or_path"])' \
    "${policy}/adapter_config.json")"
  mkdir -p "${metrics_dir}"

  for specification in \
    "open|supervised|${TRAIN_TRACES}|42" \
    "closed|supervised|${TRAIN_TRACES}|42" \
    "open|unsupervised|${ALT_TRACES}|43" \
    "closed|unsupervised|${ALT_TRACES}|43"; do
    IFS='|' read -r mode supervision traces seed <<< "${specification}"
    output="${metrics_dir}/watermark_${mode}_${supervision}.json"
    [[ -f "${output}" ]] && continue
    (
      cd "${REPO_ROOT}"
      "${SFT_CONDA_ENV_PREFIX}/bin/accelerate" launch \
        --config_file "${REPO_ROOT}/accelerate_config.yaml" \
        --num_processes "${NUM_GPUS}" \
        stages/stage4_watermark_eval.py \
        --traces "${traces}" \
        --hash-config "${HASH_CONFIG}" \
        --teacher-model "${TEACHER_MODEL}" \
        --teacher-dtype bfloat16 \
        --teacher-pad-token "" \
        --student-model "${student_model}" \
        --student-dtype bfloat16 \
        --student-pad-token "" \
        --lora-dir "${policy}" \
        --mode "${mode}" \
        --supervision "${supervision}" \
        --output "${output}" \
        --batch-size "${BATCH_SIZE}" \
        --seed "${seed}" \
        --dataset science
    )
  done
done < <(
  find "${OPD_ROOT}/exports" -mindepth 2 -maxdepth 2 -type d -name policy | sort -V
)
