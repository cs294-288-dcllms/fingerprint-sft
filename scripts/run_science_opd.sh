#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ARM_CONFIG="${1:?usage: run_science_opd.sh ARM_CONFIG [COMMON_CONFIG] [OPD_COMMON_CONFIG]}"
COMMON_CONFIG="${2:-${REPO_ROOT}/configs/science-qwen35.env}"
OPD_COMMON_CONFIG="${3:-${REPO_ROOT}/configs/opd/common.env}"
export REPO_ROOT

set -a
source "${COMMON_CONFIG}"
source "${OPD_COMMON_CONFIG}"
source "${ARM_CONFIG}"
set +a

export HF_HOME
export HF_DATASETS_CACHE
export XDG_CACHE_HOME

CKPT_DIR="${OPD_OUTPUT_DIR}/checkpoints"
EXPORT_DIR="${OPD_OUTPUT_DIR}/exports"
LOG_DIR="${OPD_OUTPUT_DIR}/logs"
EVAL_DIR="${OPD_OUTPUT_DIR}/utility_evals"
STATUS_FILE="${OPD_OUTPUT_DIR}/status"
COMPLETE_FILE="${OPD_OUTPUT_DIR}/complete"

mkdir -p "${OPD_OUTPUT_DIR}" "${CKPT_DIR}" "${EXPORT_DIR}" "${LOG_DIR}" "${EVAL_DIR}"

if [[ -f "${COMPLETE_FILE}" ]]; then
  echo "OPD arm already complete: ${OPD_RUN_NAME}"
  exit 0
fi

write_status() {
  local state="$1"
  local detail="${2:-}"
  local temporary="${STATUS_FILE}.tmp"
  {
    printf 'state=%s\n' "${state}"
    printf 'detail=%s\n' "${detail}"
    printf 'teacher_mode=%s\n' "${OPD_TEACHER_MODE}"
    printf 'updated_utc=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  } > "${temporary}"
  mv "${temporary}" "${STATUS_FILE}"
}

for required in \
  "${OPD_STUDENT_ADAPTER}/adapter_config.json" \
  "${OPD_TEACHER_ADAPTER}/adapter_config.json" \
  "${SKYRL_CONDA_ENV_PREFIX}/bin/python"; do
  [[ -e "${required}" ]] || { echo "Missing required artifact: ${required}" >&2; exit 2; }
done

if [[ "${OPD_TEACHER_MODE}" == "adfp" ]]; then
  [[ -f "${OPD_HASH_CONFIG}" ]] || {
    echo "Missing ADFP hash config: ${OPD_HASH_CONFIG}" >&2
    exit 2
  }
fi

write_status preparing "preparing prompts and merged initialization models"
if [[ ! -f "${OPD_DATA_DIR}/manifest.json" ]]; then
  "${SKYRL_CONDA_ENV_PREFIX}/bin/python" \
    "${REPO_ROOT}/scripts/prepare_skyrl_science_opd.py" \
    --train-jsonl "${SCIENCE_TRAIN_PATH}" \
    --eval-jsonl "${SCIENCE_EVAL_PATH}" \
    --output-dir "${OPD_DATA_DIR}"
fi

"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/merge_lora_checkpoint.py" \
  --base-model "${STUDENT_MODEL}" \
  --adapter "${OPD_STUDENT_ADAPTER}" \
  --output "${OPD_STUDENT_MERGED}"
"${CONDA_ENV_PREFIX}/bin/python" "${REPO_ROOT}/scripts/merge_lora_checkpoint.py" \
  --base-model "${TEACHER_MODEL}" \
  --adapter "${OPD_TEACHER_ADAPTER}" \
  --output "${OPD_TEACHER_MERGED}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export TOKENIZERS_PARALLELISM=false
export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-fingerprint-opd}"
export RAY_CGRAPH_get_timeout="${RAY_CGRAPH_get_timeout:-1800}"
export PYTHONPATH="${REPO_ROOT}:${SKYRL_DIR}${PYTHONPATH:+:${PYTHONPATH}}"

entrypoint="examples.train.on_policy_distillation.main_on_policy_distill"
if [[ "${OPD_TEACHER_MODE}" == "adfp" ]]; then
  entrypoint="scripts.main_on_policy_distill_adfp"
  export SKYRL_ADFP_TEACHER_ENABLED=1
  export SKYRL_ADFP_TEACHER_PATH="${OPD_TEACHER_MERGED}"
  export SKYRL_ADFP_PROXY_MODEL="${PROXY_MODEL}"
  export SKYRL_ADFP_HASH_CONFIG="${OPD_HASH_CONFIG}"
  export SKYRL_ADFP_LAMBDA="${OPD_ADFP_LAMBDA}"
  export SKYRL_ADFP_POSITION_CHUNK="${OPD_ADFP_POSITION_CHUNK}"
else
  unset SKYRL_ADFP_TEACHER_ENABLED
fi

write_status running "${OPD_MAX_TRAINING_STEPS} OPD steps"
(
  cd "${SKYRL_DIR}"
  "${SKYRL_CONDA_ENV_PREFIX}/bin/python" -m "${entrypoint}" \
    "data.train_data=['${OPD_DATA_DIR}/train.parquet']" \
    "data.val_data=['${OPD_DATA_DIR}/validation.parquet']" \
    trainer.algorithm.advantage_estimator=no_op \
    trainer.algorithm.policy_loss_type=importance_sampling \
    trainer.algorithm.use_kl_in_reward=true \
    trainer.algorithm.use_kl_loss=false \
    "trainer.policy.model.path=${OPD_STUDENT_MERGED}" \
    "trainer.ref.model.path=${OPD_TEACHER_MERGED}" \
    trainer.policy.model.lora.rank="${OPD_LORA_RANK}" \
    trainer.policy.model.lora.alpha="${OPD_LORA_ALPHA}" \
    trainer.policy.model.lora.dropout="${OPD_LORA_DROPOUT}" \
    trainer.policy.model.lora.target_modules=all-linear \
    trainer.policy.language_model_only=true \
    trainer.ref.language_model_only=true \
    trainer.policy.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
    trainer.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap="['Qwen3_5DecoderLayer']" \
    trainer.placement.colocate_all=true \
    trainer.strategy=fsdp \
    trainer.placement.policy_num_gpus_per_node="${OPD_NUM_GPUS}" \
    trainer.placement.ref_num_gpus_per_node="${OPD_NUM_GPUS}" \
    generator.inference_engine.num_engines="${OPD_NUM_GPUS}" \
    generator.inference_engine.tensor_parallel_size=1 \
    generator.inference_engine.language_model_only=true \
    generator.inference_engine.backend=vllm \
    generator.inference_engine.run_engines_locally=true \
    generator.inference_engine.weight_sync_backend=nccl \
    generator.inference_engine.gpu_memory_utilization="${OPD_GPU_MEMORY_UTILIZATION}" \
    generator.inference_engine.enforce_eager=false \
    generator.batched=true \
    generator.n_samples_per_prompt="${OPD_N_SAMPLES_PER_PROMPT}" \
    generator.sampling_params.max_generate_length="${OPD_MAX_GENERATE_LENGTH}" \
    generator.sampling_params.temperature="${OPD_TEMPERATURE}" \
    generator.sampling_params.top_p="${OPD_TOP_P}" \
    generator.chat_template_kwargs.enable_thinking=true \
    environment.env_class=aime \
    trainer.epochs=1 \
    trainer.max_training_steps="${OPD_MAX_TRAINING_STEPS}" \
    trainer.train_batch_size="${OPD_TRAIN_BATCH_SIZE}" \
    trainer.policy_mini_batch_size="${OPD_MINI_BATCH_SIZE}" \
    trainer.micro_forward_batch_size_per_gpu=1 \
    trainer.micro_train_batch_size_per_gpu=1 \
    trainer.update_epochs_per_batch="${OPD_UPDATE_EPOCHS}" \
    trainer.remove_microbatch_padding=false \
    trainer.max_prompt_length="${OPD_MAX_PROMPT_LENGTH}" \
    trainer.policy.optimizer_config.lr="${OPD_LEARNING_RATE}" \
    trainer.policy.optimizer_config.num_warmup_steps="${OPD_WARMUP_STEPS}" \
    trainer.policy.optimizer_config.weight_decay="${OPD_WEIGHT_DECAY}" \
    trainer.eval_before_train=false \
    trainer.eval_interval=-1 \
    trainer.seed="${OPD_SEED}" \
    trainer.logger=console \
    trainer.project_name=fingerprint-science-opd \
    "trainer.run_name=${OPD_RUN_NAME}" \
    trainer.resume_mode=latest \
    "trainer.log_path=${LOG_DIR}" \
    "trainer.ckpt_path=${CKPT_DIR}" \
    trainer.ckpt_interval="${OPD_CHECKPOINT_INTERVAL}" \
    trainer.max_ckpts_to_keep="${OPD_MAX_CHECKPOINTS}" \
    "trainer.export_path=${EXPORT_DIR}" \
    trainer.hf_save_interval="${OPD_EXPORT_INTERVAL}" \
    2>&1 | tee -a "${LOG_DIR}/train.log"
)

write_status evaluating "evaluating every ${OPD_EXPORT_INTERVAL}-step export"
latest_policy=""
while IFS= read -r policy; do
  latest_policy="${policy}"
  export_step="$(basename "$(dirname "${policy}")")"
  step_eval_dir="${EVAL_DIR}/${export_step}"
  mkdir -p "${step_eval_dir}"
  if [[ ! -f "${step_eval_dir}/student.summary.json" ]]; then
    "${CONDA_ENV_PREFIX}/bin/python" -m torch.distributed.run \
      --standalone \
      --nproc_per_node="${OPD_NUM_GPUS}" \
      "${REPO_ROOT}/scripts/eval_science_mcq.py" \
      --model "${OPD_STUDENT_MERGED}" \
      --adapter "${policy}" \
      --dataset "${SCIENCE_EVAL_PATH}" \
      --max-samples "${EVAL_SAMPLES}" \
      --batch-size "${OPD_EVAL_BATCH_SIZE}" \
      --max-new-tokens "${OPD_EVAL_MAX_NEW_TOKENS}" \
      --output "${step_eval_dir}/student.jsonl"
  fi
done < <(
  find "${EXPORT_DIR}" -mindepth 2 -maxdepth 2 -type d -name policy | sort -V
)

[[ -n "${latest_policy}" ]] || {
  echo "OPD completed without a loadable policy export." >&2
  exit 1
}
printf '%s\n' "${latest_policy}" > "${OPD_OUTPUT_DIR}/latest_policy.txt"
SFT_CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX}" "${REPO_ROOT}/scripts/verify_science_opd_completion.sh" "${OPD_OUTPUT_DIR}"
touch "${COMPLETE_FILE}"
write_status complete "training and per-export utility evaluation complete"
