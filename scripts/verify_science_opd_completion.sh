#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OPD_ROOT="${1:?Usage: verify_science_opd_completion.sh OPD_ROOT}"
SFT_CONDA_ENV_PREFIX="${SFT_CONDA_ENV_PREFIX:-${ROOT_DIR}/outputs/conda/fingerprint-sft}"
EXPECTED_STEPS="${OPD_EXPECTED_EXPORT_STEPS:-25 50 75 100 125}"
VARIANTS="open_supervised closed_supervised open_unsupervised closed_unsupervised"
MANIFEST="${OPD_ROOT}/verified_complete.json"

for step in ${EXPECTED_STEPS}; do
  policy="${OPD_ROOT}/exports/global_step_${step}/policy/adapter_config.json"
  summary="${OPD_ROOT}/utility_evals/global_step_${step}/student.summary.json"
  [[ -f "${policy}" ]] || {
    echo "Missing OPD policy export for global step ${step}: ${policy}" >&2
    exit 1
  }
  [[ -f "${summary}" ]] || {
    echo "Missing science evaluation for global step ${step}: ${summary}" >&2
    exit 1
  }
  "${SFT_CONDA_ENV_PREFIX}/bin/python" -c \
    'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("samples")==1000; assert isinstance(d.get("accuracy"), (int,float))' \
    "${summary}"
  for variant in ${VARIANTS}; do
    metric="${OPD_ROOT}/fingerprint_evals/global_step_${step}/watermark_${variant}.json"
    [[ -f "${metric}" ]] || {
      echo "Missing fingerprint evaluation for global step ${step}: ${metric}" >&2
      exit 1
    }
    mode="${variant%%_*}"
    supervision="${variant#*_}"
    "${SFT_CONDA_ENV_PREFIX}/bin/python" -c \
      'import json,sys; d=json.load(open(sys.argv[1])); assert d.get("mode")==sys.argv[2]; assert d.get("supervision")==sys.argv[3]; assert d.get("dataset")=="science"; assert int(d.get("trace_examples",0))>0; assert int(d.get("num_measurements",0))>0; assert 0.0<=float(d["mean"])<=1.0' \
      "${metric}" "${mode}" "${supervision}"
  done
done

"${SFT_CONDA_ENV_PREFIX}/bin/python" -c \
  'import json,sys; from datetime import datetime,timezone; p=sys.argv[1]; steps=[int(x) for x in sys.argv[2:]]; json.dump({"verified_utc":datetime.now(timezone.utc).isoformat(),"steps":steps,"samples_per_eval":1000,"fingerprint_variants":["open_supervised","closed_supervised","open_unsupervised","closed_unsupervised"]},open(p,"w"),indent=2); open(p,"a").write("\n")' \
  "${MANIFEST}" ${EXPECTED_STEPS}
