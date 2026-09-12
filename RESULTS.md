# Science results

All utility numbers use the fixed 1,000-question held-out split. Teacher traces
use the fixed 9,000 training prompts; fresh-seed tests regenerate the same
prompts with seed 43. ADFP uses `gamma=0.5` and hash seed `294288`. Detection is
positive when the one-sided Hoeffding p-value is below `0.05`.

## Utility

| Model/checkpoint | Accuracy | Parse rate |
|---|---:|---:|
| Qwen3.5-9B base teacher | 62.4% | 85.1% |
| Qwen3.5-9B science-SFT teacher | **85.4%** | 96.8% |
| Qwen3.5-4B base student | 57.5% | — |
| Qwen3.5-4B control SFT | **79.0%** | 92.5% |
| Qwen3.5-4B ADFP λ16 SFT | 67.3% | 78.8% |
| Qwen3.5-4B ADFP λ256 SFT | 19.6% | 34.9% |

## Teacher-trace quality

| Traces | Examples | Seed | Max new tokens | Raw accuracy | Answer-forced accuracy |
|---|---:|---:|---:|---:|---:|
| Control | 9,000 | 42 | 3,840 | 90.4% | 8.3% |
| Control | 1,000 | 43 | 3,840 | 89.8% | 7.4% |
| ADFP λ16 | 9,000 | 42 | 3,840 | 87.6% | 15.0% |
| ADFP λ16 | 1,000 | 43 | 3,840 | 87.9% | 13.4% |
| ADFP λ256 | 9,000 | 42 | 512 | 13.6% | 46.2% |
| ADFP λ256 | 1,000 | 43 | 512 | 12.9% | 43.9% |
| ADFP λ256 long-context pilot | 256 | 42 | 3,840 | 35.9% | 35.9% |

The λ256 long-context pilot missed the 70% teacher-utility gate, so no
long-context 9,000-example retraining or λ256 OPD was launched.

## Fingerprint transfer after SFT

Known-trace tests use the seed-42 training traces. Fresh-trace tests use the
same prompts regenerated with teacher seed 43.

| Student | Known open | Known closed | Fresh open | Fresh closed |
|---|---|---|---|---|
| Control SFT | 50.026%, p=0.899, no | 49.989%, p=1.000, no | 50.056%, p=0.882, no | 50.068%, p=0.835, no |
| ADFP λ16 SFT | 50.198%, p=9.93e-4, **yes** | 50.160%, p=0.0106, **yes** | 50.096%, p=0.665, no | 50.222%, p=0.113, no |
| ADFP λ256 SFT | 52.089%, p=1.47e-230, **yes** | 52.074%, p=2.09e-227, **yes** | 50.441%, p=0.00457, **yes** | 50.446%, p=0.00397, **yes** |

Repeating the λ16 fresh-seed test on all 9,000 prompts remained negative:
open 50.076% (`p=0.357`) and closed 50.046% (`p=0.690`).

## Control-teacher OPD from the λ16 student

This is the completed 125-step OPD arm. Each row uses the full 1,000-question
utility evaluation and all four fingerprint tests.

| OPD step | Accuracy | Known open p | Known closed p | Fresh open p | Fresh closed p |
|---:|---:|---:|---:|---:|---:|
| 25 | 79.0% | 0.00482 **yes** | 0.0444 **yes** | 0.709 no | 0.0999 no |
| 50 | **79.7%** | 0.00528 **yes** | 0.0282 **yes** | 0.738 no | 0.0510 no |
| 75 | 79.4% | 0.00692 **yes** | 0.0224 **yes** | 0.758 no | 0.0404 **yes** |
| 100 | 79.5% | 0.00744 **yes** | 0.0243 **yes** | 0.775 no | 0.0469 **yes** |
| 125 | 78.4% | 0.00880 **yes** | 0.0241 **yes** | 0.810 no | 0.0589 no |

## Summary

- Science SFT improved the teacher from 62.4% to 85.4% and the control student
  from 57.5% to 79.0%.
- λ16 transferred a weak fingerprint on known traces, but it did not generalize
  to fresh teacher samples, including the complete 9,000-prompt seed-43 test.
- Control-teacher OPD raised λ16-student utility to 78–80% while preserving
  known-trace detection; fresh open detection stayed negative.
- λ256 produced strong detection but unusable student utility. Increasing the
  teacher generation limit from 512 to 3,840 tokens improved
  pilot raw trace accuracy from 13.6% to 35.9%, still below the 70% training gate.
