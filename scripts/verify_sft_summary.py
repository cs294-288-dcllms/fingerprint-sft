#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONDITIONS = ("control", "ads-lambda16")
VARIANTS = (
    "open_supervised",
    "closed_supervised",
    "open_unsupervised",
    "closed_unsupervised",
)


def read_json(path: Path) -> dict[str, Any]:
    assert path.is_file(), f"missing required artifact: {path}"
    return json.loads(path.read_text(encoding="utf-8"))


def p_value(mean: float, count: int, gamma: float) -> float:
    if count <= 0 or mean <= gamma:
        return 1.0
    return max(1e-300, math.exp(-2.0 * count * (mean - gamma) ** 2))


def validate(
    summary_path: Path,
    pair_report_path: Path,
    *,
    train_seed: int,
    alt_seed: int,
    train_examples: int,
    alt_examples: int,
    eval_samples: int,
) -> dict[str, Any]:
    summary = read_json(summary_path)
    pair = read_json(pair_report_path)
    gamma = float(summary["gamma"])
    alpha = float(summary["detection_alpha"])
    assert math.isfinite(gamma)
    assert alpha == 0.05

    conditions = summary["conditions"]
    assert all(condition in conditions for condition in CONDITIONS)
    manifest_conditions: dict[str, Any] = {}
    base_accuracies: set[float] = set()

    for condition in CONDITIONS:
        condition_result = conditions[condition]
        utility = condition_result["utility"]
        assert utility["samples"] == eval_samples
        accuracy = float(utility["accuracy"])
        assert math.isfinite(accuracy)

        adapter = Path(utility["adapter"])
        predictions = Path(utility["predictions"])
        assert (adapter / "adapter_config.json").is_file(), adapter
        assert predictions.is_file(), predictions

        utility_dir = predictions.parent
        utility_summary = read_json(utility_dir / "student.summary.json")
        comparison = read_json(utility_dir / "comparison_to_base.json")
        assert utility_summary["samples"] == eval_samples
        assert float(utility_summary["accuracy"]) == accuracy
        assert comparison["samples"] == eval_samples
        assert comparison["passed"] is True
        base_accuracy = float(comparison["base_accuracy"])
        candidate_accuracy = float(comparison["candidate_accuracy"])
        assert candidate_accuracy == accuracy
        assert candidate_accuracy > base_accuracy
        base_accuracies.add(base_accuracy)

        fingerprints = condition_result["fingerprints"]
        artifact_name = adapter.parent.name
        detection: dict[str, bool] = {}
        p_values: dict[str, float] = {}
        for variant in VARIANTS:
            result = fingerprints[variant]
            raw_metric = read_json(
                summary_path.parent / "metrics" / artifact_name / f"watermark_{variant}.json"
            )
            for key in ("mean", "num_measurements", "seed", "trace_examples", "trace_file"):
                assert result[key] == raw_metric[key], (condition, variant, key)

            expected_seed = train_seed if variant.endswith("_supervised") else alt_seed
            expected_examples = train_examples if variant.endswith("_supervised") else alt_examples
            assert int(result["seed"]) == expected_seed
            assert int(result["trace_examples"]) == expected_examples
            assert Path(result["trace_file"]).is_file()

            score = p_value(float(result["mean"]), int(result["num_measurements"]), gamma)
            assert abs(float(result["p_value"]) - score) < 1e-12
            detected = score < alpha
            assert result["detected_at_0_05"] is detected
            detection[variant] = detected
            p_values[variant] = score

        if condition == "control":
            assert not any(detection.values()), detection
        else:
            assert detection["open_supervised"], detection
            assert detection["closed_supervised"], detection

        manifest_conditions[condition] = {
            "samples": eval_samples,
            "accuracy": accuracy,
            "base_accuracy": base_accuracy,
            "absolute_gain": accuracy - base_accuracy,
            "detection": detection,
            "p_values": p_values,
        }

    assert len(base_accuracies) == 1, base_accuracies
    assert pair["paired_prompts"] == alt_examples
    assert pair["prompt_mismatches"] == 0
    assert pair["solution_mismatches"] == 0
    assert pair["baseline_teacher_seed"] == train_seed
    assert pair["resampled_teacher_seed"] == alt_seed
    assert pair["identical_teacher_responses"] + pair["different_teacher_responses"] == alt_examples
    assert pair["different_teacher_responses"] > 0
    assert Path(pair["baseline_trace_file"]).is_file()
    assert Path(pair["resampled_trace_file"]).is_file()

    return {
        "verified_utc": datetime.now(UTC).isoformat(),
        "sft_training_verified": True,
        "utility_gates_passed": True,
        "fingerprint_evaluations_verified": True,
        "same_prompt_teacher_seed_pair_verified": True,
        "conditions": manifest_conditions,
        "same_prompt_teacher_seed_pair": pair,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--pair-report", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--alt-seed", type=int, required=True)
    parser.add_argument("--train-examples", type=int, required=True)
    parser.add_argument("--alt-examples", type=int, required=True)
    parser.add_argument("--eval-samples", type=int, required=True)
    args = parser.parse_args()

    payload = validate(
        args.summary,
        args.pair_report,
        train_seed=args.train_seed,
        alt_seed=args.alt_seed,
        train_examples=args.train_examples,
        alt_examples=args.alt_examples,
        eval_samples=args.eval_samples,
    )
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
