# Entropy laundering: results

Every student is Qwen3.5-4B + LoRA (r32/α64, LR 5e-5, 1 epoch, 4,096-token sequences, one seed) trained on the same
9,000 teacher traces per scheme (`traces/`), with a share of response tokens dropped from the SFT loss.

* **Detection**: open-weight radioactivity test on the first 3,000 *training* traces (teacher forcing, student top-1
  prediction scored with the key, v1 context dedup). TextSeal / SynthID: `p unweighted / entropy-weighted`. ADFP:
  its own stage-4 test (mean green-list probability, Hoeffding p, "supervised" setting). p < 0.01 = detected.
* **Accuracy**: 1,000 held-out science MCQs (`data/science/eval_1k.jsonl`), Qwen3.5 thinking-mode sampling
  (T 1.0, top-p 0.95, top-k 20, presence penalty 1.5), 2,048 new tokens. 95% CI ≈ ±2.3 points.
* Machine-readable: `reported/detection_results.csv`, `reported/accuracy_results.csv`, raw JSON in `reported/raw/`.

## Main table (entropy from Qwen3.5-4B, the student's base model)

Cells: p-value(s) (accuracy). **Bold** = fingerprint no longer detected by either test.

| Tokens dropped | TextSeal | SynthID | ADFP λ = 64 |
|---|---|---|---|
| None | 4.6e-9 / 3.3e-10 (83.2%\*) | 9.4e-7 / 6.8e-15 (82.9%) | 0.051 (55.9%) |
| Random 50% | 1.5e-5 / 6.4e-6 (82.0%) | 3.3e-4 / 5.5e-10 (82.9%) | — |
| Random 90% | **0.024 / 0.093** (83.3%) | 0.08 / 1.4e-4 (83.7%) | — |
| Lowest-entropy 50% | 5.2e-9 / 3.7e-11 (82.9%) | 9.7e-7 / 7.7e-16 (83.0%) | — |
| Highest-entropy 25% | 9.4e-4 / 0.053 (82.5%) | 0.012 / 1.0e-3 (83.0%) | — |
| Highest-entropy 35% | — | **0.15 / 0.058** (82.1%) | — |
| **Highest-entropy 50%** | **0.22 / 0.83** (81.4%) | **0.62 / 0.30** (82.4%) | **0.93** (73.9%) |
| *Never-watermarked control student* | *0.36 / 0.85 (85.1%\*)* | *0.88 / 0.40 (85.1%\*)* | *1.0 (85.1%\*)* |

\* Team's sampled run of the same eval (`/ephemeral/codex-textseal/sampled4096/`, different seed). Untrained base
student: 56.2% (2,048 tokens); 66.7% ± 5.3 at 8,192 tokens (first 300 questions).

ADFP note: the fingerprint barely transfers even without masking (mean green probability 0.5015 vs control 0.4993),
and the degraded λ = 64 traces (31.5% close `</think>`, 20% loopy) cut accuracy to 55.9%; masking the high-entropy
tokens raised it to 73.9% (finish rate 35% → 64%).

## Cross-family reference models (special tokens such as `</think>` protected from masking)

| Tokens dropped | Entropy measured by | TextSeal | SynthID |
|---|---|---|---|
| Highest-entropy 50% | Llama-3.2-3B-Instruct | 0.0042 / 0.079 (66.2%†) | 0.015 / 9.2e-5 (68.1%†) |
| Highest-entropy 50% | OLMo-3-7B-Think | 4.6e-4 / 0.0092 (81.7%) | 0.012 / 8.2e-5 (81.7%) |
| Highest-entropy 75% | OLMo-3-7B-Think | **0.14 / 0.59** (83.6%) | **0.62 / 0.20** (80.2%‡) |

† Llama is uncertain on '.', '\n\n', 'Wait', so masking makes the student verbose (answers that finish are still ~91%
correct); at 4,096 new tokens the TextSeal student reaches 75.2%. Runs without special-token protection: TextSeal
0.0065 / 0.11 (69.0%), SynthID 0.017 / 1.3e-4 (66.8%). ‡ Only borderline accuracy drop (McNemar p = 0.05).
A DeepSeek-R1-Distill-Llama-8B reference was discarded: its entropies looked broken (median 1.6 nats, correlation 0.21
with Qwen).

## Related studies on the same node

**Greedy vs sampled evaluation** (team's 1,024-token-preprocessing students): greedy showed control 81.6% vs TextSeal
72.3%; with Qwen's recommended sampling the gap disappears (80.5% vs 80.2%). The greedy gap came from greedy-decoding
loops, not from the watermark.

**Single-key TextSeal (α = 0, plain Gumbel-max) students** (`alpha0_study/`, 4,096-token training):

| Student | Training traces | p unweighted / weighted | Accuracy | Free-generation (black-box) p |
|---|---|---|---:|---:|
| α = 0, unfiltered | 9,000 | 2.4e-4 / 2.8e-5 | 83.6% | 0.087 |
| α = 0, repetition-filtered (TextSeal App. E.4 filter) | filtered subset | 8.2e-4 / 2.2e-4 | 83.7% | 0.016 |
| Control scored with the α = 0 keys | — | 0.58 / 0.80 | — | 0.89 |
| Team α = 0.1, 1,024-token | 9,000 | 1.7e-4 / 3.5e-3 | — | — |
| Team α = 0.1, 4,096-token | 9,000 | 6.6e-5 / 1.6e-3 | — | — |

The free-generation column scores the students' own eval answers (no training data needed): weak at best.
