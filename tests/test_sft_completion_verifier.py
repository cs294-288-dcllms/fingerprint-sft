from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_config(path: Path, experiment_dir: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f'CONDA_ENV_PREFIX="{Path(sys.executable).parent.parent}"',
                f'EXPERIMENT_DIR="{experiment_dir}"',
                'STUDENT_TAG_OVERRIDE="student"',
                'LEARNING_RATE="5e-5"',
                'EPOCHS="1"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifacts(experiment_dir: Path, *, passed: bool) -> None:
    adapter = (
        experiment_dir
        / "models"
        / "student_ads-lambda16_lr5e-05_e1"
        / "student_lora"
    )
    evaluation = experiment_dir / "utility_evals" / "ads-lambda16"
    adapter.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (evaluation / "student.summary.json").write_text(
        json.dumps({"samples": 1000, "accuracy": 0.7}) + "\n",
        encoding="utf-8",
    )
    (evaluation / "comparison_to_base.json").write_text(
        json.dumps(
            {
                "samples": 1000,
                "base_accuracy": 0.6,
                "candidate_accuracy": 0.7,
                "absolute_gain": 0.1,
                "passed": passed,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _run(repo_root: Path, config: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(repo_root / "scripts" / "verify_science_sft_completion.sh"), str(config)],
        text=True,
        capture_output=True,
        check=False,
    )


def test_sft_verifier_records_a_passing_utility_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=True)

    result = _run(repo_root, config)

    assert result.returncode == 0, result.stderr
    manifest = json.loads(
        (
            experiment_dir
            / "utility_evals"
            / "ads-lambda16"
            / "verified_complete.json"
        ).read_text()
    )
    assert manifest["utility_gate_passed"] is True
    assert manifest["absolute_gain"] == 0.1


def test_sft_verifier_rejects_a_failed_utility_gate(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    experiment_dir = tmp_path / "experiment"
    config = tmp_path / "science.env"
    _write_config(config, experiment_dir)
    _write_artifacts(experiment_dir, passed=False)

    result = _run(repo_root, config)

    assert result.returncode != 0
    assert not (
        experiment_dir
        / "utility_evals"
        / "ads-lambda16"
        / "verified_complete.json"
    ).exists()
