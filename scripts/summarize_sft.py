#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CONDITIONS = ("control", "ads-lambda16")
UTILITY_DIRS = {
    "control": "teachergold-control",
    "ads-lambda16": "teachergold-ads-lambda16",
}
VARIANTS = (
    "open_supervised",
    "closed_supervised",
    "open_unsupervised",
    "closed_unsupervised",
)
LABELS = {
    "open_supervised": "white-box / known traces",
    "closed_supervised": "black-box / known traces",
    "open_unsupervised": "white-box / independent traces",
    "closed_unsupervised": "black-box / independent traces",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def p_value(mean: float, count: int, gamma: float) -> float:
    if count <= 0 or mean <= gamma:
        return 1.0
    return max(1e-300, math.exp(-2.0 * count * (mean - gamma) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--student-tag", required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    args = parser.parse_args()

    gamma = float(read_json(args.exp_dir / "hash_seed" / "hash_config.json")["gamma"])
    lr_tag = f"{args.lr:g}"
    result: dict[str, Any] = {"gamma": gamma, "detection_alpha": 0.05, "conditions": {}}
    lines = [
        "# SFT fingerprint evaluation",
        "",
        "| Condition | Science accuracy | White-box known | Black-box known | White-box independent | Black-box independent |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        artifact = f"{args.student_tag}_{condition}_lr{lr_tag}_e{args.epochs}"
        utility = read_json(args.exp_dir / "utility_evals" / UTILITY_DIRS[condition] / "student.summary.json")
        metrics: dict[str, Any] = {}
        values = []
        for variant in VARIANTS:
            metric = read_json(args.exp_dir / "metrics" / artifact / f"watermark_{variant}.json")
            score = p_value(float(metric["mean"]), int(metric["num_measurements"]), gamma)
            metrics[variant] = {
                **metric,
                "p_value": score,
                "detection_alpha": 0.05,
                "detected_at_0_05": score < 0.05,
                "label": LABELS[variant],
            }
            values.append(score)
        result["conditions"][condition] = {"utility": utility, "fingerprints": metrics}
        lines.append(
            f"| {condition} | {float(utility['accuracy']):.3%} | "
            + " | ".join(
                f"{'DETECTED' if value < 0.05 else 'not detected'} (p={value:.3e})"
                for value in values
            )
            + " |"
        )
    (args.exp_dir / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (args.exp_dir / "summary.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
