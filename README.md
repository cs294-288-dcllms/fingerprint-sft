# Fingerprint Distillation

Reproducible code, configs, and the fixed science split for testing whether an ADFP fingerprint transfers from a Qwen3.5-9B teacher to a Qwen3.5-4B student through SFT and on-policy distillation.

Core ADFP stages are adapted from `YixuanEvenXu/antidistillation-fingerprinting` at commit `a05ad2bf6624b6c7d1ecf8064349f6e71035fc1e`. OPD uses SkyRL at commit `02a2b53a4142d07a38abf67f9ed7840522ee16ed`.

Completed experiment results are summarized in [`RESULTS.md`](RESULTS.md).

## Current results

The paper-style ADFP λ256 science experiment uses seed-42 fingerprinted
teacher traces for training and independently sampled seed-43 traces over the
same 9,000 prompts for cross-seed detection.

| Student SFT seed | Open-weight ln p | Open p | Closed-weight ln p | Closed p | Science accuracy |
|---:|---:|---:|---:|---:|---:|
| 42 | -5.506 | 0.00406 | -6.601 | 0.00136 | 17.5% |
| 43 | -5.554 | 0.00387 | -6.214 | 0.00200 | 17.8% |
| 44 | -5.581 | 0.00377 | -7.768 | 0.000423 | 17.9% |
| 45 | -5.632 | 0.00358 | -6.989 | 0.000922 | 17.2% |
| 46 | -5.403 | 0.00450 | -6.434 | 0.00161 | 16.8% |
| 47 | -5.575 | 0.00379 | -7.962 | 0.000349 | 17.5% |
| 48 | -5.682 | 0.00341 | -6.026 | 0.00242 | 19.2% |
| 49 | -5.591 | 0.00373 | -6.798 | 0.00112 | 17.6% |
| 50 | -5.673 | 0.00344 | -7.290 | 0.000683 | 16.7% |
| 51 | -5.370 | 0.00465 | -7.085 | 0.000837 | 17.6% |
| **Mean ± 1.96×SEM** | **-5.557 ± 0.065** | **10/10 below 0.01** | **-6.917 ± 0.393** | **10/10 below 0.01** | **17.6% ± 0.4** |

For comparison, the ADFP paper reports different-seed means of
`-4.013 ± 1.054` open-weight and `-3.478 ± 1.206` closed-weight over 10
student runs. The complete 10-run science experiment reproduces the paper's
central different-seed detectability result in this Qwen3.5 adaptation, but
not its utility behavior.

| Completed utility measurement | Accuracy |
|---|---:|
| Qwen3.5-9B control teacher traces, answer-forced | 65.41% |
| Qwen3.5-9B ADFP λ256 traces, answer-forced | 46.18% |
| Qwen3.5-4B base student | 57.5% |
| Matched paper-hyperparameter control student | 33.0% |
| ADFP λ256 students, 10-run mean | 17.6% |

The fingerprint transfers reliably across teacher sampling seeds, while λ256
substantially reduces both teacher-trace and student accuracy in this
Qwen3.5 science setting. See [`RESULTS.md`](RESULTS.md) for known-trace
measurements, controls, GTP values, λ16 results, and OPD results.

## Experiment plan

1. Fine-tune Qwen3.5-9B on 9,000 Mixture-of-Thoughts science reasoning traces and validate it on the fixed 1,000-question split.
2. Generate ADFP λ16 teacher traces and SFT Qwen3.5-4B on those traces.
3. Start both OPD arms from the same λ16 student checkpoint:
   - control teacher: ordinary science-SFT Qwen3.5-9B logits;
   - fingerprinted teacher: the same teacher with ADFP λ16 applied online to its logits.
4. At OPD steps 25, 50, 75, 100, and 125, evaluate science utility plus all four ADFP detectors (white/black-box × exact seed-42 traces and same-prompt seed-43 teacher samples).
5. On the final SFT and OPD checkpoints, repeat the same-prompt teacher test with seeds 43, 44, 45, and 46 without retraining the student.


## Data

- `data/science/train_9k.jsonl`: 9,000 science examples with reasoning traces.
- `data/science/eval_1k.jsonl`: 1,000 held-out questions with no prompt overlap.

## Setup

```bash
./scripts/bootstrap_conda.sh
./scripts/prepare_science.sh
```

Train and validate the teacher:

```bash
./scripts/run_teacher_sft.sh
```

Train the λ16 SFT student:

```bash
./scripts/run_condition.sh configs/strategies/adfp-lambda16.env
```

The OPD launcher requires this checkpoint to complete a 1,000-question evaluation and significantly outperform the base student before either OPD arm can start.

Create the separate SkyRL environment:

```bash
./scripts/setup_skyrl_opd_env.sh
```

Run both OPD arms sequentially:

```bash
./scripts/run_opd_plan.sh
```

Each stage is resumable and writes generated checkpoints, logs, and evaluations under ignored `outputs/`.

After OPD, evaluate sensitivity to the teacher's sampling seed:

```bash
./scripts/run_teacher_seed_robustness.sh
```

This reuses the same 9,000 training prompts for every seed and writes a compact
white-box/black-box table under `teacher_seed_robustness/results.md`.

## OPD configuration

| Parameter | Value |
|---|---:|
| Student initialization | Qwen3.5-4B ADFP λ16 SFT |
| Teacher | Qwen3.5-9B science SFT |
| Training prompts | 9,000 |
| Held-out evaluation prompts | 1,000 |
| Prompt batch size | 72 |
| Rollouts per prompt | 4 |
| Trajectories per update | 288 |
| Training steps | 125 |
| Policy mini-batch size | 72 |
| Control per-GPU micro-batch | 4 forward / 4 train |
| ADFP-teacher per-GPU micro-batch | 2 forward / 2 train |
| Update epochs per batch | 1 |
| Learning rate | `1e-5` |
| Warmup steps | 5 |
| Weight decay | `0.01` |
| LoRA | rank 32, alpha 64, dropout 0 |
| Sampling | temperature 1.0, top-p 1.0 |
| Maximum prompt / generation | 1,024 / 2,048 tokens |
| vLLM engine context | 4,096 tokens |
| vLLM execution mode | eager (stable eight-engine startup) |
| GPUs | 8 |
| Policy exports | every 25 steps |
| Full checkpoints | every 10 steps, keep 2 |
| Per-export evaluation | 1,000 science questions + four ADFP detector variants (exact seed-42 traces and same-prompt seed-43 samples) |

The OPD reward is the dense token-level teacher/student log-probability difference. Stored teacher responses are not OPD targets: the student generates fresh reasoning traces and the teacher scores those generated tokens.

For the fingerprinted-teacher arm:

| ADFP parameter | Value |
|---|---:|
| λ | 16 |
| γ | 0.5 |
| Hash seed | 294288 |
| Proxy model | Qwen3.5-4B |
| Context | token bigram |
| Position chunk | 8 |

Configuration files:

- `configs/science-qwen35.env`: shared models, data, and SFT settings.
- `configs/strategies/adfp-lambda16.env`: λ16 SFT student.
- `configs/strategies/adfp-paper-lambda256.env`: paper-style λ256 SFT settings.
- `configs/opd/common.env`: shared OPD hyperparameters.
- `configs/opd/control-teacher.env`: ordinary-teacher OPD arm.
- `configs/opd/adfp-teacher-lambda16.env`: online ADFP-teacher OPD arm.
- `configs/opd/adfp-teacher-lambda256.env`: λ256 ADFP-teacher OPD settings; use only after the teacher-utility and student-utility gates pass.

## Tests

```bash
make test
```
