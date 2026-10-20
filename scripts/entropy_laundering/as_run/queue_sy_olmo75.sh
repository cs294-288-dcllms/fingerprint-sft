#!/usr/bin/env bash
# SynthID OLMo-guided 75% on GPUs 4,5 after the SynthID OLMo 50% chain is done.
cd "$(dirname "$0")"
export CHAIN_EXTRA=--protect-special
until grep -q "^chain synthid-top-entropy-olmo50p done" logs/stages.log; do sleep 120; done
./chain_budget.sh 4 5 29911 synthid /ephemeral/synthid-run/synthid-train/traces.jsonl entropy/synthid-olmo.npz 0.75 olmo75p
