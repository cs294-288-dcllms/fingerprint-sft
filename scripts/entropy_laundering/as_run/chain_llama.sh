#!/usr/bin/env bash
# Wait for the Llama-reference entropy file and for this GPU pair's 25% training, then train the
# Llama-guided high-entropy 50% student and run detection + sampled eval.
# Usage: chain_llama.sh GPU_A GPU_B PORT SCHEME TRACES
set -uo pipefail
ga=$1 gb=$2 port=$3 scheme=$4 traces=$5
cd "$(dirname "$0")"
until [ -e entropy/$scheme-llama.npz ]; do sleep 120; done
until grep -q "^$scheme-top-entropy-25 train done" logs/stages.log; do sleep 120; done
./chain_budget.sh $ga $gb $port $scheme $traces entropy/$scheme-llama.npz 0.5 llama50
