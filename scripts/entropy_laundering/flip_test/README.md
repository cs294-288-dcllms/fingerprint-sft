# Does entropy predict which positions the watermark flipped?

A position is **flipped** if the watermarked trace token differs from the teacher's own top choice there (for
distortion-free schemes the key is the only source of randomness, so a non-top token means the key steered the
choice). `compute_flips.py` runs the teacher (Qwen3.5-9B + science adapter) over the traces with the generation-time
distribution (T 0.7, top-p 0.95); `analyze_flips.py` joins the labels with the Qwen3.5-4B reference entropies used by
entropy laundering. Run on the first 3,000 TextSeal training traces (4.46M positions): `textseal-analysis.json`.

| Metric | Value |
|---|---:|
| Flip rate (theory for distortion-free: mean 1 − p_max = 16.2%) | 16.3% |
| AUROC, reference-model entropy (what the attacker has) | 0.845 |
| AUROC, teacher entropy (oracle) | 0.892 |
| Flips inside the top-50% (laundering) mask | **95.2%** |
| Flip rate of kept vs masked positions | 1.5% vs 31.2% |
| Flip rate by reference-entropy decile (low → high) | 0.03, 0.03, 0.2, 1.5, 5.9, 14.6, 23.4, 30.3, 38.5, 48.7% |
| Correlation of reference and teacher entropy | 0.85 |
