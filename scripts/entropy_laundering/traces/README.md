# Teacher trace sets (gzipped JSONL, one {"prompt", "response", ...} row per training prompt)

Copied from the training node before losing access. All are Qwen3.5-9B science-teacher (LoRA, `reproduction/teacher`)
answers to the 9,000 `data/science/train_9k.jsonl` prompts, temperature 0.7, top-p 0.95, 3,840 new tokens, seed 42, no
quality filtering. Students in `../RESULTS.md` were trained on these files; detection uses the first 3,000 rows.

| File | Scheme | Original location on the node |
|---|---|---|
| `control-train9k.jsonl.gz` | none (control) | `/ephemeral/codex-textseal/reproduction/control/traces.jsonl` |
| `textseal-alpha0.1-train9k.jsonl.gz` | TextSeal α = 0.1, k = 3, keys 294288 / 306633 | `/ephemeral/codex-textseal/reproduction/textseal/traces.jsonl` |
| `synthid-train9k.jsonl.gz` | SynthID, ngram 5, depth 30 | `/ephemeral/synthid-run/synthid-train/traces.jsonl` |
| `adfp-lambda64-train9k.jsonl.gz` | ADFP (ADS) λ = 64, proxy Qwen3.5-4B, w = 2, γ = 0.5, seed 294288 | `~harshit/textseal-evals/adfp/traces/lambda64/traces.jsonl` |
| `adfp-lambda256-partial912.jsonl.gz` | ADFP λ = 256, first 912 traces only (stopped: degenerate) | `~harshit/textseal-evals/adfp/traces/lambda256-partial/` |

`SHA256SUMS` covers the gzipped files. Uncompressed SHA-256 of the ADFP λ = 64 file: see `../reported/raw/adfp-lambda64-metadata.json`.
