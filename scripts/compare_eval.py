#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Paired comparison of two evaluation files.")
    parser.add_argument("base", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-gain", type=float, default=0.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_predictions(path: Path) -> dict[int, dict[str, Any]]:
    predictions = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            index = row.get("index")
            if not isinstance(index, int) or not isinstance(row.get("correct"), bool):
                raise ValueError(f"{path}:{line_number}: missing index/correct")
            predictions[index] = row
    return predictions


def one_sided_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0 or improved <= regressed:
        return 1.0
    numerator = sum(math.comb(discordant, k) for k in range(improved, discordant + 1))
    return numerator / (2**discordant)


def main() -> None:
    args = parse_args()
    base = load_predictions(args.base)
    candidate = load_predictions(args.candidate)
    if set(base) != set(candidate):
        raise SystemExit("evaluation files do not contain the same example indices")

    indices = sorted(base)
    base_correct = sum(base[index]["correct"] for index in indices)
    candidate_correct = sum(candidate[index]["correct"] for index in indices)
    improved = sum(
        not base[index]["correct"] and candidate[index]["correct"] for index in indices
    )
    regressed = sum(
        base[index]["correct"] and not candidate[index]["correct"] for index in indices
    )
    samples = len(indices)
    base_accuracy = base_correct / samples
    candidate_accuracy = candidate_correct / samples
    gain = candidate_accuracy - base_accuracy
    p_value = one_sided_mcnemar_p(improved, regressed)
    passed = gain > args.min_gain and p_value <= args.alpha

    payload = {
        "samples": samples,
        "base_accuracy": base_accuracy,
        "candidate_accuracy": candidate_accuracy,
        "absolute_gain": gain,
        "improved_examples": improved,
        "regressed_examples": regressed,
        "one_sided_mcnemar_exact_p": p_value,
        "min_gain": args.min_gain,
        "alpha": args.alpha,
        "passed": passed,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
