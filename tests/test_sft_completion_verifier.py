from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

VARIANTS = (
    "open_supervised",
    "closed_supervised",
    "open_unsupervised",
    "closed_unsupervised",
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


def _write_config(path: Path, experiment_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'CONDA_ENV_PREFIX="{Path(sys.executable).parent.parent}"',
                f'EXPERIMENT_DIR="{experiment_dir}"',
                'TRAIN_SEED="42"',
                'ALT_SEED="43"',
                'NUM_EXAMPLES="9000"',
                'ALT_NUM_EXAMPLES="1000"',
                'EVAL_SAMPLES="1000"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifacts(
    experiment_dir: Path,
    *,
    passed: bool,
    detect_known_lambda16: bool = True,
    prompt_mismatches: int = 0,
) -> None:
    conditions = {}
    trace_dir = experiment_dir / "traces"
    trace_dir.mkdir(parents=True)

    baseline_trace = trace_dir / "baseline.jsonl"
    resampled_trace = trace_dir / "resampled.jsonl"
    baseline_trace.write_text("{}\n", encoding="utf-8")
    resampled_trace.write_text("{}\n", encoding="utf-8")

    for condition, accuracy in (("control", 0.7), ("ads-lambda16", 0.65)):
        artifact_name = f"student_{condition}_lr5e-05_e1"
        adapter = experiment_dir / "models" / artifact_name / "student_lora"
        adapter.mkdir(parents=True)
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")

        utility_dir = experiment_dir / "utility_evals" / condition
        predictions = utility_dir / "student.jsonl"
        predictions.parent.mkdir(parents=True)
        predictions.write_text("{}\n", encoding="utf-8")
        _write_json(
            utility_dir / "student.summary.json",
            {"samples": 1000, "accuracy": accuracy},
        )
        _write_json(
            utility_dir / "comparison_to_base.json",
            {
                "samples": 1000,
                "base_accuracy": 0.6,
                "candidate_accuracy": accuracy,
                "absolute_gain": accuracy - 0.6,
                "passed": passed if condition == "ads-lambda16" else True,
            },
        )

        fingerprints = {}
        for variant in VARIANTS:
            supervised = variant.endswith("_supervised")
            should_detect = condition == "ads-lambda16" and supervised and detect_known_lambda16
            mean = 0.52 if should_detect else 0.5
            count = 10000
            score = 0.000335462627902512 if should_detect else 1.0
            raw_metric = {
                "mean": mean,
                "num_measurements": count,
                "seed": 42 if supervised else 43,
                "trace_examples": 9000 if supervised else 1000,
                "trace_file": str(baseline_trace if supervised else resampled_trace),
            }
            _write_json(
                experiment_dir / "metrics" / artifact_name / f"watermark_{variant}.json",
                raw_metric,
            )
            fingerprints[variant] = {
                **raw_metric,
                "p_value": score,
                "detection_alpha": 0.05,
                "detected_at_0_05": should_detect,
            }

        conditions[condition] = {
            "utility": {
                "adapter": str(adapter),
                "predictions": str(predictions),
                "samples": 1000,
                "accuracy": accuracy,
            },
            "fingerprints": fingerprints,
        }

    _write_json(
        experiment_dir / "summary.json",
        {"gamma": 0.5, "detection_alpha": 0.05, "conditions": conditions},
    )
    _write_json(
        experiment_dir / "metrics" / "same_prompt_teacher_seed_pair.json",
        {
            "paired_prompts": 1000,
            "prompt_mismatches": prompt_mismatches,
            "solution_mismatches": 0,
            "identical_teacher_responses": 0,
            "different_teacher_responses": 1000,
            "baseline_teacher_seed": 42,
            "resampled_teacher_seed": 43,
            "baseline_trace_file": str(baseline_trace),
            "resampled_trace_file": str(resampled_trace),
        },
    )


def _run(repo_root: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo_root / "scripts" / "verify_science_sft_completion.sh"), str(config)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_sft_verifier_records_complete_training_and_evaluation(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=True)

    result = _run(repo_root, config)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((experiment_dir / "sft_verified_complete.json").read_text())
    assert manifest["utility_gates_passed"] is True
    assert manifest["fingerprint_evaluations_verified"] is True
    assert manifest["conditions"]["ads-lambda16"]["detection"]["closed_supervised"]


def test_sft_verifier_rejects_a_failed_utility_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=False)

    result = _run(repo_root, config)

    assert result.returncode != 0
    assert not (experiment_dir / "sft_verified_complete.json").exists()


def test_sft_verifier_rejects_missing_known_trace_detection(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=True, detect_known_lambda16=False)

    result = _run(repo_root, config)

    assert result.returncode != 0
    assert not (experiment_dir / "sft_verified_complete.json").exists()


def test_sft_verifier_rejects_prompt_mismatches(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=True, prompt_mismatches=1)

    result = _run(repo_root, config)

    assert result.returncode != 0
    assert not (experiment_dir / "sft_verified_complete.json").exists()
