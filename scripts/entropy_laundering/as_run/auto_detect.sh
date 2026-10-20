#!/usr/bin/env bash
# Run detection for each token-drop student as soon as its training finishes.
set -uo pipefail
cd "$(dirname "$0")"
TS=/ephemeral/codex-textseal/reproduction/textseal/traces.jsonl
SY=/ephemeral/synthid-run/synthid-train/traces.jsonl
jobs=(synthid-drop50:synthid:$SY
      textseal-drop90:textseal:$TS synthid-drop90:synthid:$SY)
pending=("${jobs[@]}")
while [ ${#pending[@]} -gt 0 ]; do
  next=()
  for job in "${pending[@]}"; do
    IFS=: read -r name scheme traces <<< "$job"
    if grep -q "^$name train done" logs/stages.log 2>/dev/null; then
      ./run_detect.sh ${DETECT_GPUS:-6,7} 29741 "$name:students/$name:$scheme:$traces"
    else
      next+=("$job")
    fi
  done
  pending=("${next[@]}")
  [ ${#pending[@]} -gt 0 ] && sleep 120
done
echo "auto_detect finished $(date -u +%FT%TZ)"
