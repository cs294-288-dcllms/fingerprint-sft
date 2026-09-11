from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


@pytest.mark.parametrize("expected_examples", [1000, 9000])
def test_multi_seed_summary_verifies_same_prompts_and_metrics(
    tmp_path: Path, expected_examples: int
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    results = tmp_path / "results"
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    result_suffix = (
        "" if expected_examples == 1000 else f"_n{expected_examples}"
    )
    write_json(
        results / "targets" / "sft" / "target.json",
        {"label": "λ16 SFT", "adapter": str(adapter)},
    )
    for seed in (43, 44):
        trace = tmp_path / f"seed{seed}.jsonl"
        trace.write_text("{}\n", encoding="utf-8")
        write_json(
            results / "pairs" / f"seed{seed}{result_suffix}.json",
            {
                "paired_prompts": expected_examples,
                "prompt_mismatches": 0,
                "solution_mismatches": 0,
                "different_teacher_responses": expected_examples,
                "baseline_teacher_seed": 42,
                "resampled_teacher_seed": seed,
                "prompt_sha256": "a" * 64,
                "resampled_trace_file": str(trace),
            },
        )
        for mode in ("open", "closed"):
            write_json(
                results
                / "targets"
                / "sft"
                / f"seed{seed}{result_suffix}"
                / f"watermark_{mode}.json",
                {
                    "dataset": "science",
                    "mode": mode,
                    "supervision": "unsupervised",
                    "seed": seed,
                    "trace_examples": expected_examples,
                    "trace_file": str(trace),
                    "student_lora_dir": str(adapter),
                    "num_measurements": 100,
                    "mean": 0.5,
                    "gamma": 0.5,
                    "p_value": 1.0,
                    "detected_at_0_05": False,
                },
            )

    output = results / "verified_complete.json"
    markdown = results / "results.md"
    completed = subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "summarize_teacher_seed_robustness.py"),
            "--results-root",
            str(results),
            "--seeds",
            "43",
            "44",
            "--expected-examples",
            str(expected_examples),
            "--output",
            str(output),
            "--markdown",
            str(markdown),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    summary = json.loads(output.read_text(encoding="utf-8"))
    assert summary["teacher_sampling_seeds"] == [43, 44]
    assert summary["paired_prompts"] == expected_examples
    assert summary["student_retrained_per_seed"] is False
    assert summary["targets"]["sft"]["modes"]["open"]["detected_seed_count"] == 0
    assert "Seed 43 p / detected" in markdown.read_text(encoding="utf-8")
    assert f"same {expected_examples:,} prompts" in markdown.read_text(encoding="utf-8")
