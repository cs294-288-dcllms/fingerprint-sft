#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMMON_CONFIG="${1:-${REPO_ROOT}/configs/science-qwen35.env}"
export REPO_ROOT

set -a
source "${COMMON_CONFIG}"
set +a

LR_TAG="$("${CONDA_ENV_PREFIX}/bin/python" -c \
  'import sys; print(f"{float(sys.argv[1]):g}")' "${LEARNING_RATE}")"
ADAPTER="${EXPERIMENT_DIR}/models/${STUDENT_TAG_OVERRIDE}_ads-lambda16_lr${LR_TAG}_e${EPOCHS}/student_lora/adapter_config.json"
EVAL_DIR="${EXPERIMENT_DIR}/utility_evals/ads-lambda16"
SUMMARY="${EVAL_DIR}/student.summary.json"
COMPARISON="${EVAL_DIR}/comparison_to_base.json"
MANIFEST="${EVAL_DIR}/verified_complete.json"

for required in "${ADAPTER}" "${SUMMARY}" "${COMPARISON}"; do
  [[ -f "${required}" ]] || {
    echo "Missing required λ16 SFT artifact: ${required}" >&2
    exit 1
  }
done

"${CONDA_ENV_PREFIX}/bin/python" -c \
  'import json,math,sys
summary=json.load(open(sys.argv[1]))
comparison=json.load(open(sys.argv[2]))
assert summary.get("samples")==1000, summary
accuracy=summary.get("accuracy")
assert isinstance(accuracy,(int,float)) and math.isfinite(accuracy), summary
assert comparison.get("samples")==1000, comparison
assert comparison.get("passed") is True, comparison
base=comparison.get("base_accuracy")
candidate=comparison.get("candidate_accuracy")
assert isinstance(base,(int,float)) and isinstance(candidate,(int,float)), comparison
assert candidate > base, comparison
assert abs(candidate-accuracy) < 1e-12, (summary,comparison)' \
  "${SUMMARY}" "${COMPARISON}"

"${CONDA_ENV_PREFIX}/bin/python" -c \
  'import json,sys
from datetime import datetime,timezone
summary=json.load(open(sys.argv[1]))
comparison=json.load(open(sys.argv[2]))
payload={
  "verified_utc":datetime.now(timezone.utc).isoformat(),
  "samples":summary["samples"],
  "accuracy":summary["accuracy"],
  "base_accuracy":comparison["base_accuracy"],
  "absolute_gain":comparison["absolute_gain"],
  "utility_gate_passed":True,
}
with open(sys.argv[3],"w") as handle:
  json.dump(payload,handle,indent=2)
  handle.write("\n")' \
  "${SUMMARY}" "${COMPARISON}" "${MANIFEST}"
