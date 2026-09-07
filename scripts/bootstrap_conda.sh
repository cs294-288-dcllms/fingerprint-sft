#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONDA_BIN="${CONDA_BIN:-${HOME}/miniconda3/bin/conda}"
CONDA_ENV_PREFIX="${CONDA_ENV_PREFIX:-/tmp/fingerprint-sft-conda}"
CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-/tmp/fingerprint-conda-pkgs}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/fingerprint-pip-cache}"

export CONDA_PKGS_DIRS PIP_CACHE_DIR

if [[ ! -x "${CONDA_BIN}" ]]; then
  echo "Conda not found at ${CONDA_BIN}" >&2
  exit 1
fi
if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --prefix "${CONDA_ENV_PREFIX}" python=3.12 pip
fi
"${CONDA_ENV_PREFIX}/bin/python" -m pip install --upgrade pip
"${CONDA_ENV_PREFIX}/bin/python" -m pip install --requirement "${REPO_ROOT}/requirements-gpu.txt"
echo "Environment ready: ${CONDA_ENV_PREFIX}"
