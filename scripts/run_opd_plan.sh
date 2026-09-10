#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

"${REPO_ROOT}/scripts/run_science_opd.sh" \
  "${REPO_ROOT}/configs/opd/control-teacher.env"
"${REPO_ROOT}/scripts/run_science_opd.sh" \
  "${REPO_ROOT}/configs/opd/adfp-teacher-lambda16.env"
