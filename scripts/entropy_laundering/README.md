# Entropy laundering: removing distillation fingerprints by loss masking

The student is trained on watermarked teacher traces, but a chosen fraction of response tokens is dropped
from the SFT loss (labels set to -100; the tokens stay in the input as context). Choosing the tokens a
reference model is most uncertain about removes the TextSeal and SynthID fingerprints at little accuracy
cost, because the watermark signal concentrates in high-entropy positions. The attacker needs only
the traces: no teacher access and no key. Results: [RESULTS.md](RESULTS.md). **Start with [HANDOVER.md](HANDOVER.md)** (status, setup, caveats, next steps, file map);
background notes in [docs/CONCEPTS.md](docs/CONCEPTS.md); trace data in [traces/](traces/); slide figures in [figures/](figures/).

## Files

| File | Purpose |
|---|---|
| `sft_tokendrop.py` | Wraps `stages/stage3_finetune.py`; masks response tokens from the loss. `--drop-rate`, `--mask-mode {random,top-entropy,bottom-entropy,top-surprisal}`, `--entropy-file`, `--protect-special` (never mask special tokens such as `</think>`/EOS). Also sets TRL `max_length` to `--max-seq-length` (TRL 0.24 otherwise truncates at 1,024). Writes a per-rank audit of the kept fraction. |
| `compute_entropy.py` | Per-token entropy/surprisal of the traces under a same-tokenizer reference model (Qwen3.5-4B base). |
| `compute_entropy_xref.py` | Same for a reference model with a different tokenizer (Llama, OLMo); maps scores to student tokens by character offsets (`--mapping max\|start`). |
| `eval_science_mcq_sampled.py` | MCQ accuracy with Qwen3.5 thinking-mode sampling (T 1.0, top-p 0.95, top-k 20, presence penalty 1.5). |
| `run_drop.sh` | Train students: `run_drop.sh GPUS PORT NAME:TRACES:RATE ...` (extra flags via `EXTRA_ARGS`). |
| `run_entropy.sh`, `run_entropy_xref.sh` | Entropy passes (`REF=<model dir>` for the cross-family one). |
| `run_detect.sh` | TextSeal/SynthID open-weight radioactivity on the first 3,000 traces (`NAME:ADAPTER:SCHEME:TRACES`). |
| `run_detect_adfp.sh` | ADFP detection via `stages/stage4_watermark_eval.py`. |
| `run_eval.sh` | Sampled accuracy eval on the 1,000 held-out questions. |
| `chain_entropy.sh`, `chain_budget.sh` | Train → detect → eval chains for entropy-targeted masking. |

## Environment

* `FP_ROOT` (default `/ephemeral/codex-textseal`): provides `venv/`, `cache/huggingface` and `reproduction/hash.json`.
* `DETECTOR_CODE` (default `/ephemeral/synthid-run/code`): a checkout of `scripts/eval_textseal_radioactivity.py`
  with `--scheme synthid` support; the copy on `main` is TextSeal-only.
* Outputs (`students/`, `entropy/`, `detection*/`, `results/`, `logs/`) are written next to the scripts and git-ignored.

## Example

```bash
# Entropy of the TextSeal traces under the Qwen reference model
./run_entropy.sh 0 /path/to/textseal/traces.jsonl entropy/textseal.npz
# Mask the 50% highest-entropy response tokens, then detect and evaluate
EXTRA_ARGS="--mask-mode top-entropy --entropy-file entropy/textseal.npz" \
  ./run_drop.sh 4,5 29500 textseal-top-entropy-50:/path/to/textseal/traces.jsonl:0.5
NPROC=1 ./run_detect.sh 4 29505 textseal-top-entropy-50:students/textseal-top-entropy-50:textseal:/path/to/textseal/traces.jsonl
./run_eval.sh 5 textseal-top-entropy-50
```
