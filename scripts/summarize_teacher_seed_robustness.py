#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def expected_p_value(mean: float, count: int, gamma: float) -> float:
    if count <= 0 or mean <= gamma:
        return 1.0
    return max(1e-300, math.exp(-2.0 * count * (mean - gamma) ** 2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify and summarize same-prompt teacher-seed robustness."
    )
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--expected-examples", type=int, default=1000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()

    seeds = list(dict.fromkeys(args.seeds))
    if len(seeds) < 2:
        raise SystemExit("at least two teacher sampling seeds are required")
    if args.expected_examples <= 0:
        raise SystemExit("expected examples must be positive")
    result_suffix = (
        "" if args.expected_examples == 1000 else f"_n{args.expected_examples}"
    )

    prompt_digest: str | None = None
    pairs: dict[str, dict[str, Any]] = {}
    for seed in seeds:
        pair = read_json(
            args.results_root / "pairs" / f"seed{seed}{result_suffix}.json"
        )
        assert pair["paired_prompts"] == args.expected_examples
        assert pair["prompt_mismatches"] == 0
        assert pair["solution_mismatches"] == 0
        assert pair["baseline_teacher_seed"] == 42
        assert pair["resampled_teacher_seed"] == seed
        assert pair["different_teacher_responses"] > 0
        if prompt_digest is None:
            prompt_digest = pair["prompt_sha256"]
        assert pair["prompt_sha256"] == prompt_digest
        pairs[str(seed)] = pair

    target_dirs = sorted(
        path for path in (args.results_root / "targets").iterdir() if path.is_dir()
    )
    if not target_dirs:
        raise SystemExit("no evaluated targets found")

    targets: dict[str, Any] = {}
    markdown = [
        "# Same known prompts, different teacher sampling seeds",
        "",
        (
            f"Every row uses the same {args.expected_examples:,} prompts. "
            "Only the teacher decoding seed changes; no student is "
            "retrained."
        ),
        "",
        "| Checkpoint | Detector | "
        + " | ".join(f"Seed {seed} p / detected" for seed in seeds)
        + " | Detected seeds |",
        "|---|---|" + "|".join(["---:"] * (len(seeds) + 1)) + "|",
    ]

    for target_dir in target_dirs:
        metadata = read_json(target_dir / "target.json")
        adapter = str(Path(metadata["adapter"]).resolve())
        target_result: dict[str, Any] = {
            "label": metadata["label"],
            "adapter": adapter,
            "modes": {},
        }
        for mode in ("open", "closed"):
            mode_result: dict[str, Any] = {}
            rendered: list[str] = []
            detected_count = 0
            for seed in seeds:
                metric = read_json(
                    target_dir
                    / f"seed{seed}{result_suffix}"
                    / f"watermark_{mode}.json"
                )
                pair = pairs[str(seed)]
                assert metric["dataset"] == "science"
                assert metric["mode"] == mode
                assert metric["supervision"] == "unsupervised"
                assert metric["seed"] == seed
                assert metric["trace_examples"] == args.expected_examples
                assert str(Path(metric["trace_file"]).resolve()) == str(
                    Path(pair["resampled_trace_file"]).resolve()
                )
                assert str(Path(metric["student_lora_dir"]).resolve()) == adapter
                count = int(metric["num_measurements"])
                mean = float(metric["mean"])
                gamma = float(metric["gamma"])
                expected = expected_p_value(mean, count, gamma)
                assert math.isclose(
                    float(metric["p_value"]),
                    expected,
                    rel_tol=1e-12,
                    abs_tol=1e-300,
                )
                detected = expected < 0.05
                assert metric["detected_at_0_05"] is detected
                detected_count += int(detected)
                mode_result[str(seed)] = {
                    "mean": mean,
                    "num_measurements": count,
                    "p_value": expected,
                    "detected_at_0_05": detected,
                }
                rendered.append(f"{expected:.3g} / {'yes' if detected else 'no'}")
            target_result["modes"][mode] = {
                "seeds": mode_result,
                "detected_seed_count": detected_count,
                "total_seed_count": len(seeds),
            }
            markdown.append(
                f"| {metadata['label']} | "
                f"{'white-box' if mode == 'open' else 'black-box'} | "
                + " | ".join(rendered)
                + f" | {detected_count}/{len(seeds)} |"
            )
        targets[target_dir.name] = target_result

    result = {
        "verified_utc": datetime.now(UTC).isoformat(),
        "baseline_teacher_seed": 42,
        "teacher_sampling_seeds": seeds,
        "paired_prompts": args.expected_examples,
        "prompt_sha256": prompt_digest,
        "student_retrained_per_seed": False,
        "pairs": pairs,
        "targets": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
