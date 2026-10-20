# Handover: entropy laundering and related follow-ups (Harshit, Oct 2026)

Written on 2026-10-20, before possibly losing access to the 8×H100 node. Everything needed to understand,
reproduce and continue the work is in this directory; what could not be copied is listed under "Not preserved".

## 1. One-paragraph summary

We distill a watermarked Qwen3.5-9B science teacher into a Qwen3.5-4B LoRA student and ask whether the watermark
("fingerprint") survives. It does for TextSeal and SynthID (p ≈ 1e-10 to 1e-15 on the training traces). **Entropy
laundering** — dropping the 50% highest-entropy response tokens (by the student's own base model) from the SFT loss,
while keeping them as context — makes both fingerprints undetectable (p = 0.3–0.8, same as a never-watermarked
student) with accuracy within noise (81.4 / 82.4% vs 83.2 / 82.9%). Dropping the *lowest*-entropy half does nothing
and dropping a random half does not remove the fingerprint, so the effect comes from *which* tokens are dropped.
The attacker needs only the traces and one forward pass of an open model: no teacher access, no key.
Results: [RESULTS.md](RESULTS.md).

## 2. Setup

| Piece | Details |
|---|---|
| Data | `data/science/` (repo root): `open-r1/Mixture-of-Thoughts` science config (revision `e55fa28`), rows from NVIDIA Llama-Nemotron post-training. 4-option MCQ with reasoning traces. `train_9k` (9,000 rows, 8,183 unique prompts), `eval_1k` (1,000 unique prompts, no exact overlap). |
| Teacher | Qwen3.5-9B + LoRA SFT on the 9,000 gold traces (1 epoch, LR 3e-5, r32/α64). 84.6% on eval_1k (greedy). Adapter on the node: `/ephemeral/codex-textseal/reproduction/teacher`. |
| Trace sets | Teacher answers the 9,000 train prompts per scheme, T 0.7, top-p 0.95, 3,840 tokens, seed 42, no filtering. Copies in `traces/`. |
| Schemes | TextSeal α = 0.1, k = 3, keys 294288 / 306633 · SynthID ngram 5, depth 30 · ADFP (ADS) λ = 64, proxy Qwen3.5-4B, w = 2, γ = 0.5 · control (none). |
| Students | Qwen3.5-4B + LoRA r32/α64/dropout 0.05, LR 5e-5, 1 epoch, batch 1 × grad-accum 8 × 2 GPUs (16 traces/step), 4,096-token sequences (TRL 0.24 `max_length` patched to 4,096 — see Gotchas). |
| Loss (laundering) | L(θ) = −(1/|K|) Σ_{(i,t)∈K} log p_θ(x_{i,t} \| x_{i,<t}), K = response tokens with H_ref ≤ τ; τ = **global** median entropy over all 13.3M training tokens (0.39 nats for TextSeal). Per trace 34–63% are dropped (5th–95th pct). Normalised by kept tokens per step (`average_tokens_across_devices=True`). Prompt never trained. |
| Detection | `run_detect.sh` → `eval_textseal_radioactivity.py` (`--scheme textseal|synthid`, first 3,000 training traces, v1 global context dedup; entropy weights = √ of the student's own min-max-normalised entropy). ADFP: `run_detect_adfp.sh` → `stages/stage4_watermark_eval.py --supervision supervised --mode open`. |
| Accuracy | `run_eval.sh` → `eval_science_mcq_sampled.py`, eval_1k, Qwen thinking sampling, 2,048 tokens. |

## 3. Key findings

1. **The inherited signal concentrates in high-entropy tokens.** Lowest-entropy 50% masking: no effect. Highest 25%:
   weakened; 35–50%: gone. Random 50%: survives (random 90% removes TextSeal but not SynthID — SynthID weighted p 1.4e-4).
   Caveat: "concentrated", not "entirely": the 25–50% band still carries some signal.
2. **Accuracy holds** with a same-family reference model (≤ 1.8 points, within ±2.3 CI, one seed). A cross-family
   reference (Llama) makes the student verbose (66%); OLMo-3-7B-Think works but needs a 75% budget.
3. **ADFP at λ = 64 barely transfers** (p = 0.051 unmasked) and its traces are degraded (accuracy 55.9%); λ = 256 was
   worse (14% of traces close `</think>`, 57% loopy) and was stopped at 912 traces. ADFP is therefore not a meaningful
   test of the attack here. Masking *raised* the ADFP student's accuracy to 73.9% (it stops learning the loops).
4. **The greedy-decoding accuracy gap** between control and TextSeal students (81.6 vs 72.3%) was a decoding artifact;
   with Qwen's sampling settings they match (80.5 vs 80.2%).
5. **Most of the distillation gain is conciseness**: the untrained base student's *finished* answers are already ~81%
   correct; it mainly fails by running out of tokens (56.2% at 2,048 tokens, 66.7% at 8,192).

## 4. Caveats (state these when presenting)

* One training seed per student; borderline p-values are noisy.
* Detection is the most favourable setting for the owner: teacher forcing on the student's *own training traces*,
  with the student's weights. We never tested fresh-text detection on laundered students.
* Only in-distribution accuracy (eval_1k comes from the same pool). No OOD benchmark yet (GPQA-Diamond / MMLU-Pro STEM
  proposed; needs a download and, for GPQA, a gated-dataset account).
* One task (science MCQ), one model pair (Qwen3.5-9B → 4B).
* The direct evidence that the signal sits in high-entropy positions is interventional (masking). A per-entropy-bin
  breakdown of the unmasked student's detection scores would show it directly (detector change needed: it only saves totals).

## 5. Open next steps (rough priority)

1. **Gradient / teacher-contrast detector on laundered students.** A teammate's learned detector
   (`/ephemeral/codex-textseal/two-key1024-repo/experiments/alpha0_alignment1024_final_report.md`): score
   ⟨Tw − Tc, ∇ log p_T0(y|x)⟩ / |y| on the student's fresh answers, Tw/Tc = teacher fine-tuned on its own
   watermarked / clean traces; AUROC 0.996 vs 0.69 for key-based TextSeal on α = 0 students; the cheap
   log p_Tw − log p_Tc variant separated perfectly. Running it (or the likelihood-ratio variant) on the
   highest-entropy-50% students is the strongest test of laundering. Needs Tw/Tc trained on our trace sets.
2. OOD accuracy (GPQA-Diamond, MMLU-Pro STEM) for base, control, unmasked and laundered students.
3. Seeds (≥ 3) for the key rows: none, random 50%, highest-entropy 50%.
4. Per-entropy-bin detection-score breakdown (see caveats).
5. Optional: per-trace-median threshold ablation (expected ≈ same: both rules drop every token above 1 nat).

## 6. Where things are

In this directory (`scripts/entropy_laundering/`):

| Path | Contents |
|---|---|
| `sft_tokendrop.py`, `compute_entropy*.py`, `eval_science_mcq_sampled.py`, `run_*.sh`, `chain_*.sh` | Portable code (paths via `FP_ROOT`, `DETECTOR_CODE`); see README.md |
| `adfp/gen_lambda.sh` | ADFP trace generation as run |
| `as_run/` | One-off queue/orchestration scripts exactly as run on the node (hard-coded node paths) |
| `alpha0_study/` | α = 0 (single-key) student study scripts |
| `traces/` | Gzipped trace sets (control, TextSeal, SynthID, ADFP λ64, ADFP λ256 partial) + SHA256SUMS |
| `reported/` | CSVs of every detection / accuracy number, raw summary JSONs, token-drop audits (kept fraction), stage timelines |
| `figures/` | Slide figures (matplotlib scripts + PNG/SVG); `figures/data/trace0_tokens.json` from `extract_trace_tokens.py` |
| `docs/blog.html` | Paper-style blog draft (MathJax) on the project, from Oct 15 (before the laundering results) |
| `docs/CONCEPTS.md` | Plain-language notes: watermark schemes, detection, p-values, related work with section pointers, slide text |

**Not preserved (node only):** student adapters (~267 MB each, `~harshit/textseal-evals/tokendrop/students/`,
`.../alpha0/students/`), entropy files (`.../tokendrop/entropy/*.npz`, regenerate with `run_entropy.sh` in
~10–20 min), per-question eval outputs (`.../results/*.jsonl`, 317 MB), detection work arrays. The teacher adapter
and team students live under `/ephemeral/codex-textseal/`. If the adapters matter, upload them to the HF Hub before
access ends — at least `textseal-top-entropy-50`, `synthid-top-entropy-50`, `textseal-drop0`, `synthid-drop0`.

## 7. Reproducing the main rows

```bash
cd scripts/entropy_laundering
export FP_ROOT=/path/with/venv-and-hf-cache DETECTOR_CODE=/path/to/synthid-capable/detector-checkout
zcat traces/textseal-alpha0.1-train9k.jsonl.gz > /tmp/textseal.jsonl
./run_entropy.sh 0 /tmp/textseal.jsonl entropy/textseal.npz                       # Qwen3.5-4B reference
EXTRA_ARGS="--mask-mode top-entropy --entropy-file entropy/textseal.npz" \
  ./run_drop.sh 0,1 29500 textseal-top-entropy-50:/tmp/textseal.jsonl:0.5          # ~1–1.5 h on 2 H100
NPROC=1 ./run_detect.sh 0 29505 textseal-top-entropy-50:students/textseal-top-entropy-50:textseal:/tmp/textseal.jsonl
./run_eval.sh 1 textseal-top-entropy-50
```
SynthID detection needs a detector checkout with `--scheme synthid` (teammate's `synthid-text` branch; the copy on
`main` is TextSeal-only). ADFP: `adfp/gen_lambda.sh`, then `run_detect_adfp.sh`.

## 8. Gotchas we hit

* **TRL 0.24 truncates at 1,024 tokens** (`SFTConfig.max_length` default) even when preprocessing allows 4,096; the
  team's early students were trained on truncated traces. `sft_tokendrop.py` sets `max_length=4096` and errors if any
  example would be truncated.
* **Dedup v2 is invalid** (it flags never-watermarked controls); use v1 (`global_context_dedup_v1`).
* **Greedy eval is misleading** for these students (loops); use Qwen's sampling settings.
* **Cross-tokenizer reference models** need character-offset mapping (`compute_entropy_xref.py --mapping max|start`)
  and special-token protection (`--protect-special`), or `</think>`/EOS get masked.
* The node's root disk fills fast (HF caches, triton caches); GPUs 0–3 belonged to another group (short jobs only).
* ADFP traces take ~9 h for 9,000 prompts on 4 GPUs; students on these long traces train in ~70 min on 2 GPUs.
