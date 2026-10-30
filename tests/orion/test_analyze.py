"""Check reporting denominators and preservation of unknown outcome semantics."""

import json
import tempfile
import unittest
from pathlib import Path

from opd_pilot.analyze import run, summarize
from opd_pilot.records import Recorder


class Analysis(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.records = self.root / "records"

    def write(self, outcomes):
        with Recorder(self.records, "worker-1") as writer:
            for index, (task, outcome) in enumerate(outcomes):
                writer.record(
                    "episode", episode_uid=f"episode-{index}", task_id=task, outcome=outcome
                )

    def test_done_and_raw_reward_do_not_infer_success(self):
        self.write([("task", {"episode_done": True, "raw_score": 1})])
        result = summarize(self.records)
        self.assertEqual(result["success"]["unknown"], 1)
        self.assertIsNone(result["success_rate_known"])
        self.assertEqual(result["scores"]["raw_score"]["mean"], 1)
        self.assertIsNone(result["scores"]["score_final"]["mean"])

    def test_known_denominators_and_task_weighting(self):
        self.write(
            [
                ("a", {"task_success": True, "score_final": 30, "score_best": 80}),
                ("a", {"task_success": True}),
                ("b", {"task_success": False, "technical_failure": True}),
                ("b", {"task_success": None}),
            ]
        )
        result = summarize(self.records)
        self.assertAlmostEqual(result["success_rate_known"], 2 / 3)
        self.assertEqual(result["success_label_coverage"], 0.75)
        self.assertEqual(result["task_equal_success_rate_known"], 0.5)
        self.assertEqual(result["technical_failure"], {"yes": 1, "no": 0, "unknown": 3})
        self.assertEqual(result["scores"]["score_final"], {"mean": 30, "count": 1})
        self.assertEqual(result["scores"]["score_best"]["mean"], 80)

    def test_invalid_success_label_rejected(self):
        self.write([("task", {"task_success": "false"})])
        with self.assertRaises(ValueError):
            summarize(self.records)

    def test_report_is_separate_and_cannot_overwrite(self):
        self.write([("task", {"task_success": False})])
        output = self.root / "report"
        spec = dict(
            records_root=str(self.records),
            output=str(output),
            environment="alfworld",
            split="dev",
            model_revision="recorded-revision",
            protocol="pilot-v1",
        )
        run(spec)
        status = json.loads((output / "status.json").read_text())
        self.assertEqual(status["status"], "completed")
        self.assertTrue((output / "report.md").is_file())
        with self.assertRaises(FileExistsError):
            run(spec)
        self.assertEqual(summarize(self.records)["episodes"], 1)
