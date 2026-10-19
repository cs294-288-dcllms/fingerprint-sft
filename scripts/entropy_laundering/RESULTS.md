# Loss-masking attack on distillation fingerprints

Open-weight detection on 3,000 training traces (v1 dedup). Accuracy: sampled eval (Qwen thinking settings), 1,000 held-out questions, 2,048-token cap; * = team sampler run (different seed). One training seed each. Control accuracy 85.1%* (team run). Base student (untrained): 56.2% at 2,048 tokens; 66.7% +/- 5.3 at 8,192 tokens (300 Qs). Llama-guided TextSeal at 4,096 tokens: 75.2%. Special tokens protected for Llama/OLMo runs.

| Scheme | Masking | Reference model | p unweighted | p entropy-weighted | Accuracy |
|---|---|---|---:|---:|---:|
| TextSeal | control student (null) | — | 0.36 | 0.85 | — |
| TextSeal | none | — | 4.6e-09 | 3.3e-10 | 83.2%* |
| TextSeal | random 50% | — | 1.5e-05 | 6.4e-06 | 82.0% |
| TextSeal | random 90% | — | 0.024 | 0.093 | 83.3% |
| TextSeal | low-entropy 50% | Qwen | 5.2e-09 | 3.7e-11 | 82.9% |
| TextSeal | high-entropy 25% | Qwen | 0.00094 | 0.053 | 82.5% |
| TextSeal | high-entropy 50% | Qwen | 0.22 | 0.83 | 81.4% |
| TextSeal | high-entropy 50% | Llama-3.2-3B | 0.0042 | 0.079 | 66.2% |
| TextSeal | high-entropy 50% | Olmo-3-7B-Think | 0.00046 | 0.0092 | 81.7% |
| TextSeal | high-entropy 75% | Olmo-3-7B-Think | 0.14 | 0.59 | 83.6% |
| SynthID | control student (null) | — | 0.88 | 0.4 | — |
| SynthID | none | — | 9.4e-07 | 6.8e-15 | 82.9% |
| SynthID | random 50% | — | 0.00033 | 5.5e-10 | 82.9% |
| SynthID | random 90% | — | 0.08 | 0.00014 | 83.7% |
| SynthID | low-entropy 50% | Qwen | 9.7e-07 | 7.7e-16 | 83.0% |
| SynthID | high-entropy 25% | Qwen | 0.012 | 0.001 | 83.0% |
| SynthID | high-entropy 35% | Qwen | 0.15 | 0.058 | 82.1% |
| SynthID | high-entropy 50% | Qwen | 0.62 | 0.3 | 82.4% |
| SynthID | high-entropy 50% | Llama-3.2-3B | 0.015 | 9.2e-05 | 68.1% |
| SynthID | high-entropy 50% | Olmo-3-7B-Think | 0.012 | 8.2e-05 | 81.7% |
| SynthID | high-entropy 75% | Olmo-3-7B-Think | 0.62 | 0.2 | 80.2% |
