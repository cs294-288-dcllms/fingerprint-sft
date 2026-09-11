from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path


STEPS = (25, 50, 75, 100, 125)
VARIANTS = (
    ("open", "supervised"),
    ("closed", "supervised"),
    ("open", "unsupervised"),
    ("closed", "unsupervised"),
)


def _write_completed_step(root: Path, step: int) -> None:
    policy = root / "exports" / f"global_step_{step}" / "policy"
    evaluation = root / "utility_evals" / f"global_step_{step}"
    policy.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (policy / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (evaluation / "student.summary.json").write_text(
        json.dumps({"samples": 1000, "accuracy": 0.75}) + "\n",
        encoding="utf-8",
    )
    metrics = root / "fingerprint_evals" / f"global_step_{step}"
    metrics.mkdir(parents=True)
    for mode, supervision in VARIANTS:
        (metrics / f"watermark_{mode}_{supervision}.json").write_text(
            json.dumps(
                {
                    "dataset": "science",
                    "mode": mode,
                    "supervision": supervision,
                    "trace_examples": 1000,
                    "num_measurements": 10,
                    "mean": 0.5,
                    "gamma": 0.5,
                    "p_value": 1.0,
                    "detection_alpha": 0.05,
                    "detected_at_0_05": False,
                }
            )
            + "\n",
            encoding="utf-8",
        )


def _run_verifier(repo_root: Path, opd_root: Path) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["SFT_CONDA_ENV_PREFIX"] = str(Path(sys.executable).parent.parent)
    return subprocess.run(
        [str(repo_root / "scripts" / "verify_science_opd_completion.sh"), str(opd_root)],
        text=True,
        capture_output=True,
        env=env,
        check=False,
    )


def test_verifier_requires_and_records_all_expected_steps(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for step in STEPS:
        _write_completed_step(tmp_path, step)

    result = _run_verifier(repo_root, tmp_path)

    assert result.returncode == 0, result.stderr
    manifest = json.loads((tmp_path / "verified_complete.json").read_text())
    assert manifest["steps"] == list(STEPS)
    assert manifest["samples_per_eval"] == 1000
    assert manifest["fingerprint_variants"] == [
        "open_supervised",
        "closed_supervised",
        "open_unsupervised",
        "closed_unsupervised",
    ]


def test_verifier_fails_when_an_evaluation_is_missing(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for step in STEPS[:-1]:
        _write_completed_step(tmp_path, step)
    policy = tmp_path / "exports" / "global_step_125" / "policy"
    policy.mkdir(parents=True)
    (policy / "adapter_config.json").write_text("{}\n", encoding="utf-8")

    result = _run_verifier(repo_root, tmp_path)

    assert result.returncode != 0
    assert "Missing science evaluation for global step 125" in result.stderr


def test_verifier_fails_when_a_fingerprint_evaluation_is_missing(
    tmp_path: Path,
) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    for step in STEPS:
        _write_completed_step(tmp_path, step)
    (
        tmp_path
        / "fingerprint_evals"
        / "global_step_50"
        / "watermark_closed_unsupervised.json"
    ).unlink()

    result = _run_verifier(repo_root, tmp_path)

    assert result.returncode != 0
    assert "Missing fingerprint evaluation for global step 50" in result.stderr
