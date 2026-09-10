#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-conda}"
UV_BIN="${UV_BIN:-uv}"
CONDA_ENV_PREFIX="${SKYRL_CONDA_ENV_PREFIX:-${ROOT_DIR}/outputs/conda/skyrl-opd}"
SKYRL_DIR="${SKYRL_DIR:-${ROOT_DIR}/external/skyrl}"
SKYRL_REPOSITORY="${SKYRL_REPOSITORY:-https://github.com/NovaSky-AI/SkyRL.git}"
SKYRL_COMMIT="${SKYRL_COMMIT:-02a2b53a4142d07a38abf67f9ed7840522ee16ed}"
REQUIREMENTS="${ROOT_DIR}/requirements-opd.txt"
READY_FILE="${CONDA_ENV_PREFIX}/.skyrl-opd-ready"

if [[ ! -d "${SKYRL_DIR}/.git" ]]; then
  git clone "${SKYRL_REPOSITORY}" "${SKYRL_DIR}"
fi

actual_commit="$(git -C "${SKYRL_DIR}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${SKYRL_COMMIT}" ]]; then
  git -C "${SKYRL_DIR}" fetch origin "${SKYRL_COMMIT}"
  git -C "${SKYRL_DIR}" checkout --detach "${SKYRL_COMMIT}"
fi

SKYRL_CONFIG_PATH="${SKYRL_DIR}/skyrl/train/config/config.py"
if ! grep -q SKYRL_FORCE_EAGER_LORA "${SKYRL_CONFIG_PATH}"; then
  sed -i 's|if _uses_lora_weight_sync(self) and ie_cfg.enforce_eager and ie_cfg.backend == "vllm":|if _uses_lora_weight_sync(self) and ie_cfg.enforce_eager and ie_cfg.backend == "vllm" and os.environ.get("SKYRL_FORCE_EAGER_LORA", "0") != "1":|' "${SKYRL_CONFIG_PATH}"
fi
grep -q SKYRL_FORCE_EAGER_LORA "${SKYRL_CONFIG_PATH}" || {
  echo "Failed to patch SkyRL explicit eager-LoRA override" >&2
  exit 1
}

if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --prefix "${CONDA_ENV_PREFIX}" python=3.12 pip -y
fi

(
  cd "${ROOT_DIR}"
  "${UV_BIN}" pip install \
    --python "${CONDA_ENV_PREFIX}/bin/python" \
    --requirements "${REQUIREMENTS}"
)

"${CONDA_ENV_PREFIX}/bin/python" - <<'PY'
import ray
import skyrl
import torch
import transformers
import vllm
import vllm_router

print(
    "SkyRL OPD environment ready:",
    f"torch={torch.__version__}",
    f"transformers={transformers.__version__}",
    f"ray={ray.__version__}",
    f"vllm={vllm.__version__}",
)
PY

touch "${READY_FILE}"
