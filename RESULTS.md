# Science results

All student utility numbers use the fixed 1,000-question held-out science
split. Teacher traces use the fixed 9,000 training prompts. Different-seed
tests regenerate the same 9,000 prompts with a new teacher sampling seed.
ADFP uses `gamma=0.5`, token-bigram hashing, and hash seed `294288`.

## Utility baselines

| Model/checkpoint | Accuracy | Parse rate |
|---|---:|---:|
| Qwen3.5-9B base teacher | 62.4% | 85.1% |
| Qwen3.5-9B science-SFT teacher | **85.4%** | 96.8% |
| Qwen3.5-4B base student | 57.5% | 83.1% |
| Qwen3.5-4B control SFT, LoRA r32, LR `5e-5` | **79.0%** | 92.5% |
| Qwen3.5-4B ADFP λ16 SFT | 67.3% | 78.8% |

## ADFP λ256 with the paper's SFT hyperparameters

This is a Qwen3.5 science adaptation of the paper's λ256 experiment, not an
exact reproduction of its DeepSeek/Qwen2.5 GSM8K setting. Both the control and
ADFP students use LoRA rank/alpha 128, dropout 0.05, learning rate `1e-4`,
global batch size 8, and one epoch.

### Teacher-trace accuracy

| Traces | Sampling set | Examples | Raw accuracy | Answer-forced accuracy |
|---|---|---:|---:|---:|
| Control | Training seed | 9,000 | 34.63% | **65.41%** |
| Control | Different seed | 9,000 | 34.33% | **65.53%** |
| ADFP λ256 | Training seed | 9,000 | 13.20% | **46.18%** |
| ADFP λ256 | Different seed | 9,000 | 13.50% | **46.58%** |

λ256 reduced training-seed answer-forced teacher accuracy by 19.23 percentage
points relative to control. The two sampling seeds produced nearly identical
accuracy, so this quality loss is not a sampling-seed artifact.

### Student accuracy

| Student | Accuracy | Parse rate | Change from base |
|---|---:|---:|---:|
| Qwen3.5-4B base | 57.5% | 83.1% | — |
| Paper-hyperparameter control SFT | **33.0%** | 56.9% | -24.5 points |
| Paper-hyperparameter ADFP λ256 SFT | **17.5%** | 30.7% | -40.0 points |

The controlled ADFP-versus-control difference is -15.5 accuracy points.
Because the unfingerprinted control itself fell to 33.0%, the paper's
rank-128/`1e-4` SFT configuration is not utility-stable for this Qwen3.5
science adaptation. For comparison, the r32/`5e-5` control above reached
79.0%.

### Complete fingerprint measurements

Known-trace tests use the exact teacher traces on which the student was
fine-tuned. Different-seed tests use independently sampled teacher traces for
the same 9,000 prompts. The reported p-value is the paper's one-sided
Hoeffding test; lower is stronger evidence of fingerprint transfer.

| Student | Evaluation traces | Access | Contexts | GTP | ln p | p |
|---|---|---|---:|---:|---:|---:|
| Control | Known | Open-weight | 487,623 | 49.9085% | 0.000 | 1.000 |
| Control | Known | Closed-weight | 487,623 | 49.9476% | 0.000 | 1.000 |
| Control | Different seed | Open-weight | 488,160 | 49.8504% | 0.000 | 1.000 |
| Control | Different seed | Closed-weight | 488,160 | 49.8400% | 0.000 | 1.000 |
| ADFP λ256 | Known | Open-weight | 606,320 | 52.0831% | -526.194 | 3.00e-229 |
| ADFP λ256 | Known | Closed-weight | 606,320 | 52.0999% | -534.714 | 5.98e-233 |
| ADFP λ256 | Different seed | Open-weight | 606,251 | 50.2131% | **-5.506** | **0.00406** |
| ADFP λ256 | Different seed | Closed-weight | 606,251 | 50.2333% | **-6.601** | **0.00136** |

The different-seed fingerprint is detected in both access settings at
`p < 0.01`; every unfingerprinted control has `p = 1`. The different-seed
signal retains about 10–11% of the known-trace GTP shift, but remains
detectable because it is measured over more than 606,000 deduplicated
contexts.

### Comparison with the paper

The [ADFP paper](https://arxiv.org/abs/2602.03812) reports the following
different-seed results for its original λ256 LoRA setting. Paper values are
means over 10 student runs.

| Different-seed detector | Paper mean ln p | Qwen3.5 science, seed 42 |
|---|---:|---:|
| Open-weight | -4.013 ± 1.054 | **-5.506** |
| Closed-weight | -3.478 ± 1.206 | **-6.601** |

### Repeated student-seed validation

All students use the same seed-42 fingerprinted training traces and are
tested against the same independently sampled seed-43 teacher traces. Only
the student SFT seed changes.

| Student SFT seed | Open GTP | Open ln p | Open p | Closed GTP | Closed ln p | Closed p | Accuracy |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | 50.2131% | -5.506 | **0.00406** | 50.2333% | -6.601 | **0.00136** | 17.5% |
| 43 | 50.2140% | -5.554 | **0.00387** | 50.2264% | -6.214 | **0.00200** | 17.8% |
| 44 | 50.2145% | -5.581 | **0.00377** | 50.2531% | -7.768 | **0.000423** | 17.9% |
| 45 | 50.2155% | -5.632 | **0.00358** | 50.2401% | -6.989 | **0.000922** | 17.2% |

Different-seed detection succeeds at `p < 0.01` for both open- and
closed-weight tests in **4/4 independent student SFT runs**. The mean ln p
over these four runs is -5.568 ± 0.052 open and -6.893 ± 0.650 closed
(`1.96×SEM`), compared with the paper's -4.013 ± 1.054 and -3.478 ± 1.206.
Mean student accuracy is 17.6% ± 0.3 (`1.96×SEM`). More student seeds are running
to test whether this holds across the paper's full 10-run protocol.

These results reproduce the paper's central different-seed detectability
claim for four Qwen3.5 science students. They do not yet reproduce its full
10-run evidence or its low utility loss.

## ADFP λ16 SFT

| Evaluation | Open-weight p | Closed-weight p |
|---|---:|---:|
| Known training traces | 9.93e-4 | 0.0106 |
| Different seed, complete 9,000 prompts | 0.357 | 0.690 |

λ16 transfers a detectable fingerprint on known traces but not to the
complete different-seed teacher sample.

## Control-teacher OPD from the λ16 student

This completed 125-step arm starts from the λ16 SFT student and performs
on-policy distillation against the ordinary, unfingerprinted teacher.

| OPD step | Accuracy | Known open p | Known closed p | Different-seed open p | Different-seed closed p |
|---:|---:|---:|---:|---:|---:|
| 25 | 79.0% | 0.00482 | 0.0444 | 0.709 | 0.0999 |
| 50 | **79.7%** | 0.00528 | 0.0282 | 0.738 | 0.0510 |
| 75 | 79.4% | 0.00692 | 0.0224 | 0.758 | 0.0404 |
| 100 | 79.5% | 0.00744 | 0.0243 | 0.775 | 0.0469 |
| 125 | 78.4% | 0.00880 | 0.0241 | 0.810 | 0.0589 |

Control-teacher OPD restores λ16-student utility to approximately 78–80%
while preserving its known-trace signal. It does not produce consistent
different-seed detection.

## Conclusions

- The verified λ256 student is strongly detected on known traces.
- The λ256 fingerprint is also detected on independently sampled,
  same-prompt teacher traces in 4/4 student runs.
- Seed 42: open `p=0.00406`, closed `p=0.00136`.
- Seed 43: open `p=0.00387`, closed `p=0.00200`.
- Seed 44: open `p=0.00377`, closed `p=0.000423`.
- Seed 45: open `p=0.00358`, closed `p=0.000922`.
- Unfingerprinted controls remain undetected.
- The paper's full 10-run robustness test is still in progress.
- Utility is the unresolved gap: teacher answer-forced accuracy falls from
  65.41% to 46.18%, and the four λ256 students average 17.6% accuracy versus
  33.0% for the matched unfingerprinted control.
