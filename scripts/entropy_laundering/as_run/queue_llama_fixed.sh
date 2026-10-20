#!/usr/bin/env bash
# Llama-guided high-entropy 50% with special tokens protected, both schemes, on GPUs 6,7.
cd "$(dirname "$0")"
export CHAIN_EXTRA=--protect-special
until grep -q "^chain textseal-top-entropy-llama50 done" logs/stages.log; do sleep 120; done
./chain_budget.sh 6 7 29851 textseal /ephemeral/codex-textseal/reproduction/textseal/traces.jsonl entropy/textseal-llama.npz 0.5 llama50p
./chain_budget.sh 6 7 29861 synthid /ephemeral/synthid-run/synthid-train/traces.jsonl entropy/synthid-llama.npz 0.5 llama50p
