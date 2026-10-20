# Concepts and context (plain-language notes for slides and new readers)

Figures for each scheme are in `../figures/` (`green_red_watermark`, `adfp`, `gumbel`, `textseal`, `synthid`,
`laundering*`).

## Watermark schemes

**Green/red list** (Kirchenbauer et al., ICML 2023). At each step, hash(key, previous token(s)) splits the vocabulary
into green and red; add a bonus δ to green logits. Detect: count green tokens, z-test. Distorts the distribution
(stronger δ = stronger signal, worse text). Low-entropy text is hard to watermark (§1.2; Thm 4.2 ties strength to spike entropy).

**ADFP** (Xu et al., 2026, "Antidistillation Fingerprinting"). Green/red, but the bias is chosen to be *learnable by
the student*: z̃_t = z_t + λ·q_t(𝟙[t ∈ green] − Σ_{s∈green} q_s), with q = a proxy student's probabilities. The
q-term is the gradient of the proxy's green-probability mass w.r.t. its logits (from ⟨∇log q_t, ∇L⟩ under an isotropic
JJᵀ ≈ cI approximation). Detection: mean green probability of the student over contexts, Hoeffding p =
exp(−2n(GTP − 0.5)²). Main evaluation uses *fresh teacher samples on the same prompts* (§A.4); window w = 2.

**Gumbel-max** (Aaronson & Kirchner, 2022; formula in Fernandez et al. 2023 "Three Bricks", and Aaronson's Simons talk
slide 3, Oct 2024). R_w = hash(key, previous k tokens, w) ∈ [0, 1]; pick argmax_w R_w^{1/p_w}. Each word still wins
with probability exactly p_w over random keys (distortion-free per token; equivalent to the Gumbel-max trick /
exponential races), but with a fixed key the same context always gives the same token → less diversity, loops.
Detect: S = Σ −ln(1 − R_t) ~ Gamma(T, 1) under the null (exact p-value). R·p would *not* be distortion-free.

**TextSeal** (Sander et al., Meta, 2026). Gumbel-max with two keys: each token uses key A (1 − α) or key B (α) at
random, which breaks loops (looping 16% → 7% in our traces at α = 0.1) at a cost in per-token signal
(signal-to-noise ∝ √((1−α)² + α²): 0.91 at α = 0.1, 0.71 at α = 0.5). Detection blends both keys' scores with the same
weights (early fusion) and weights tokens by √(normalised entropy) (§3.2, Fig. 9), p-value from a moment-matched Gamma.
Radioactivity test (§6, §A.4): teacher-force the training traces through the student, score its top-1 predictions.

**SynthID-Text** (Dathathri et al., *Nature* 634:818–823, 2024, doi:10.1038/s41586-024-08025-4). Tournament sampling:
draw candidates from the model, run m knockout rounds where hash(key, context, round) gives each word g ∈ {0, 1} and
higher g wins (ties random). Distortion-free on average and still random. Detect: mean g over tokens × rounds
(binomial tail; weighted variants). Ours: 30 rounds, 5-token context.

## Detection basics

* **p-value**: probability, if the student were *not* trained on watermarked data, of evidence at least this strong.
  It is not "the probability it was a fluke". p < 0.01 is our threshold.
* **Radioactivity test (what our tables use)**: feed each training trace to the student, take its top-1 guess at each
  position, score the guess with the key and the previous k trace tokens, deduplicate repeated contexts (v1), sum,
  Gamma p-value. A control student's guesses are independent of the key → scores average 1. A watermarked student
  learned the teacher's key-tilted choices at uncertain positions → excess.
* **Needs the exact training traces for Gumbel-family schemes**: the key's numbers are pseudo-random per
  (context, token), so a student can only memorise specific continuations; fresh text shares few contexts (our
  free-generation p ≈ 0.016–0.09). ADFP's 2-token window and learnability-by-design let it transfer to fresh samples.
* **nats**: entropy with natural log (1 nat ≈ 1.44 bits); e^H ≈ effective number of choices. Cutoff 0.39 nats ≈ 1.5
  choices.

## Entropy laundering: why it works

The key can only tip a choice when there is a choice. Proposition 3 of TextSeal: E[S_T] ≥ T + (π²/6 − 1)·H_T, so the
signal scales with entropy. Removing the highest-entropy positions from the loss means the student never practises the
key-tilted choices, while it still learns the content from the near-certain tokens. A *global* entropy threshold
matches the mechanism (signal depends on absolute entropy); a per-trace threshold would give nearly the same mask here.

**Related work** (none found that does exactly this, quick search only):
* Removal of radioactivity: Pan et al., ACL 2025 (arXiv 2502.11598) — paraphrase training data or neutralise at
  inference; CDG-KD (arXiv 2504.17480) — contrastive decoding; trace-rewriting defence (arXiv 2602.15143) tests a
  regex token filter.
* Same insight on text: SIRA (Cheng et al., ICML 2025, §3.2) rewrites high self-information tokens; SWEET (Lee et
  al., ACL 2024, §3.1–3.3) and EWD use entropy for detection.
* Dropping tokens from the loss without hurting quality: Goldfish loss (Hans et al., NeurIPS 2024, §5.1–5.2: random
  1/k of tokens, k = 3–4) — our random baseline is essentially this; Rho-1 (Lin et al., 2024, §3.2–3.4: keep ~60–70% of
  tokens, entropy as a score works too).

## Teammate's gradient (teacher-contrast) detector

T0 = clean teacher; Tw / Tc = T0 fine-tuned on its own α = 0 watermarked / clean traces. Score a suspect response y to
a fresh prompt: ⟨Tw − Tc, ∇θ log p_T0(y|x)⟩ / |y| ≈ log p_Tw(y|x) − log p_Tc(y|x). Works on the student's own outputs
(no student weights, no key test). Clean subtraction matters because Tc ≠ T0: the traces were sampled at T 0.7 /
top-p 0.95 (sharper than T0), plus finite-sample and Adam drift, so Tw − T0 is dominated by generic "sharpening"
shared with any distilled student. Closest prior: GaussMark (Block, Rakhlin, Sekhari, ICML 2025; same gradient-inner-
product score, random secret direction). Caveats: learned detector (wrong-key controls needed), empirical thresholds,
α = 0 / 1,024-token setting only. Name candidates we liked: "Watermark Vectors", "Decoy Distillation".

## Slide text used

* Gumbel-max: "Doesn't distort the probabilities, but the same context always gives the same token."
* Entropy laundering: "The watermark signal concentrates in tokens where the model had a real choice. Fine-tune on the
  watermarked traces, but drop the top 50% by reference-model entropy from the loss (they stay as context). The
  fingerprint becomes undetectable (p ≈ 0.3–0.8, same as a never-watermarked student); accuracy within ~2 points.
  Needs only the traces: no teacher, no key."
