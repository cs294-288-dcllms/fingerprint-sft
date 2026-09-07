import json
import tempfile
import unittest
from pathlib import Path

from stages.stage5_plotting import P_VALUE_FLOOR, compute_pvalue, gather_points


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


class Stage5PlottingTest(unittest.TestCase):
    def test_pvalue_is_floored_for_log_scale_plotting(self) -> None:
        self.assertEqual(compute_pvalue(0.6, 1_000_000, 0.5), P_VALUE_FLOOR)

    def test_unsupervised_plot_uses_alternative_teacher_eval(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            exp_dir = Path(temp_dir) / "teacher_proxy_science_n1"
            metric_dir = exp_dir / "metrics" / "student_control_lr5e-05_e1"
            _write_json(
                metric_dir / "watermark_open_unsupervised.json",
                {"mean": 0.6, "num_measurements": 100},
            )
            _write_json(
                exp_dir / "training_traces" / "control" / "teacher_eval.json",
                {"answer_forced_accuracy": 0.1},
            )
            _write_json(
                exp_dir / "alternative_traces" / "control" / "teacher_eval.json",
                {"answer_forced_accuracy": 0.9},
            )

            buckets, _ = gather_points(
                exp_dir,
                "open_unsupervised",
                student_tag="student",
                lr="5e-05",
                epochs="1",
            )

            self.assertEqual(buckets["control"][0][0], 0.9)


if __name__ == "__main__":
    unittest.main()
