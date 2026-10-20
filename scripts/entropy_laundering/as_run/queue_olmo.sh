#!/usr/bin/env bash
# OLMo-guided high-entropy 50% (special tokens protected). TextSeal on GPUs 6,7 as soon as its
# entropy file exists; SynthID on GPUs 4,5 after the SynthID Llama-reference chain is done.
cd "$(dirname "$0")"
export CHAIN_EXTRA=--protect-special
(
  until [ -e entropy/textseal-olmo.npz ]; do sleep 60; done
  ./chain_budget.sh 6 7 29881 textseal /ephemeral/codex-textseal/reproduction/textseal/traces.jsonl entropy/textseal-olmo.npz 0.5 olmo50p
) &
(
  until [ -e entropy/synthid-olmo.npz ] && grep -q "^chain synthid-top-entropy-llama50p done" logs/stages.log; do sleep 60; done
  ./chain_budget.sh 4 5 29891 synthid /ephemeral/synthid-run/synthid-train/traces.jsonl entropy/synthid-olmo.npz 0.5 olmo50p
) &
wait
