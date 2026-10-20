#!/usr/bin/env bash
# Start the SynthID 35% high-entropy run once the SynthID Llama-reference chain is done.
cd "$(dirname "$0")"
until grep -q "^chain synthid-top-entropy-llama50 done" logs/stages.log; do sleep 120; done
./chain_budget.sh 4 5 29841 synthid /ephemeral/synthid-run/synthid-train/traces.jsonl entropy/synthid.npz 0.35 35
