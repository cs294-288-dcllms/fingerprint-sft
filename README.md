# Fingerprint SFT

Core fingerprinting stages are adapted from [YixuanEvenXu/antidistillation-fingerprinting](https://github.com/YixuanEvenXu/antidistillation-fingerprinting) at upstream commit `a05ad2bf6624b6c7d1ecf8064349f6e71035fc1e`.

Reusable code and configs for training and evaluating fingerprinted teacher-to-student SFT runs.

The default configuration uses:

- Teacher: `Qwen/Qwen3.5-9B`
- Proxy/student: `Qwen/Qwen3.5-4B`
- Data: 9,000 Mixture-of-Thoughts science training traces and 1,000 held-out questions
- Strategies: no fingerprint, ADFP λ=8/16/32, and radioactive δ=2
- Evaluation: science utility plus white-box/black-box detection on known and independent traces

## Setup

```bash
./scripts/bootstrap_conda.sh
```

The pinned environment defaults to `/tmp/fingerprint-sft-conda`. Override `CONDA_ENV_PREFIX` if needed.

## Science data

The exact deterministic split used by the experiments is included:

- `data/science/train_9k.jsonl`: 9,000 thinking-trace training examples
- `data/science/eval_1k.jsonl`: 1,000 held-out examples with no prompt overlap

To build the tokenizer-rendered teacher SFT file locally:

```bash
./scripts/prepare_science.sh
```

The source split is reused as-is; only the derived SFT file is written under ignored `data/local/`.

## Train and validate the teacher

```bash
./scripts/run_teacher_sft.sh
```

The teacher adapter is accepted only when its 1,000-question held-out evaluation significantly improves over the base teacher.

## Run one SFT strategy

```bash
./scripts/run_condition.sh configs/strategies/control.env
./scripts/run_condition.sh configs/strategies/adfp-lambda8.env
./scripts/run_condition.sh configs/strategies/adfp-lambda16.env
./scripts/run_condition.sh configs/strategies/adfp-lambda32.env
./scripts/run_condition.sh configs/strategies/radioactive-delta2.env
```

Each strategy creates a separate student LoRA checkpoint. Science generation defaults to 3,840 new tokens so long Qwen reasoning traces can reach their final answer while retaining prompt headroom in the SFT window. Compatible completed traces are used only as response-length hints to group similar examples and reduce padding waste. If generation exceeds GPU memory, the runner halves the Stage 1 batch size and resumes its per-rank checkpoints.

## Run the complete SFT matrix

```bash
./scripts/run_all_sft.sh
```

Stages are resumable through output files and sentinels. Results are written under `outputs/experiments/` and are intentionally excluded from Git.

## Tests

```bash
make test
```

## Configuration

- `configs/science-qwen35.env`: models, paths, distributed settings, and SFT hyperparameters
- `configs/teacher-sft.env`: teacher-specific SFT and evaluation settings
- `configs/strategies/*.env`: fingerprint method and strength

All values can be overridden with environment variables.
