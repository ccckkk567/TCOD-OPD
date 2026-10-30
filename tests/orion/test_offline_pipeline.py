"""Exercise pilot evidence selection, leakage rejection and task-level statistics."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_pilot.audit import check
from opd_pilot.evidence import publish, separate_output
from opd_pilot.paired_stats import compare
from opd_pilot.probes import main as probes_main
from opd_pilot.probes import select
from opd_pilot.records import Recorder


class OfflinePipeline(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.protocol = root / "protocol"
        self.protocol.mkdir()
        counts = {"train_pilot": 1, "analysis_dev": 2, "confirm_holdout": 1, "smoke": 1}
        for split, count in counts.items():
            rows = [{"task_id": f"{split}-{i}", "official_split": "train"} for i in range(count)]
            (self.protocol / f"{split}.jsonl").write_text(
                "\n".join(json.dumps(r) for r in rows) + "\n"
            )
        (self.protocol / "protocol.json").write_text(
            json.dumps(
                {
                    "actual_counts": counts,
                    "official_test_used": False,
                    "teacher_success_filter": False,
                }
            )
        )
        (self.protocol / "SPLITS_READY.json").write_text(json.dumps({"actual_counts": counts}))
        self.records = root / "records"
        with Recorder(self.records, "worker-1") as writer:
            for task in range(2):
                eid = f"episode-{task}"
                writer.record(
                    "episode", episode_uid=eid, task_id=f"analysis_dev-{task}", outcome={}
                )
                for turn in range(3):
                    writer.record(
                        "turn",
                        turn_uid=f"{eid}-turn-{turn}",
                        episode_uid=eid,
                        sampling_student_revision="student-v1",
                        scoring_teacher_revision="teacher-v1",
                        messages=[{"role": "user", "content": f"actual-{task}-{turn}"}],
                        tokens=[10, task, turn, 99],
                        prompt_length=3,
                        response="look",
                        action="look",
                        raw_reward=0,
                        student_logprobs=[-0.5],
                        teacher_logprobs=[-0.2],
                    )
        self.spec = dict(
            protocol_root=str(self.protocol),
            records_root=str(self.records),
            split="analysis_dev",
            student_revision="student-v1",
            teacher_revision="teacher-v1",
            max_prefixes=4,
            max_per_task=2,
            seed=42,
        )

    def test_fixed_pool_is_reproducible_balanced_and_exact(self):
        result = select(self.spec)
        self.assertEqual(result, select(self.spec))
        self.assertEqual(result["actual"], 4)
        tasks = [row["task_id"] for row in result["prefixes"]]
        self.assertEqual(tasks.count("analysis_dev-0"), 2)
        self.assertEqual(tasks.count("analysis_dev-1"), 2)
        for row in result["prefixes"]:
            self.assertEqual(len(row["prefix_tokens"]), 3)
            self.assertNotIn(99, row["prefix_tokens"])
            self.assertTrue(row["messages"][0]["content"].startswith("actual-"))
            self.assertFalse(row["environment_replay_validated"])

    def test_no_confirmation_or_revision_substitution(self):
        with self.assertRaises(ValueError):
            select(dict(self.spec, split="confirm_holdout"))
        with self.assertRaises(ValueError):
            check(dict(self.spec, split="train_pilot"))
        with self.assertRaisesRegex(ValueError, "revision"):
            check(dict(self.spec, teacher_revision="different-teacher"))

    def test_split_overlap_is_rejected(self):
        (self.protocol / "smoke.jsonl").write_text(
            json.dumps(
                {
                    "task_id": "analysis_dev-0",
                    "official_split": "train",
                }
            )
            + "\n"
        )
        with self.assertRaisesRegex(ValueError, "overlapping"):
            check(self.spec)

    def test_publish_cannot_replace_or_write_into_evidence(self):
        with self.assertRaises(ValueError):
            separate_output(self.records / "output", self.records)
        output = self.records.parent / "bank"
        publish(output, {"bank.json": select(self.spec)})
        with self.assertRaises(FileExistsError):
            publish(output, {"bank.json": {}})

    def test_cli_validation_then_publication(self):
        output = self.records.parent / "bank"
        spec_path = self.records.parent / "spec.json"
        spec_path.write_text(json.dumps(dict(self.spec, output=str(output))))
        with patch("sys.argv", ["probes", "--spec", str(spec_path), "--validate-only"]):
            with contextlib.redirect_stdout(io.StringIO()):
                probes_main()
        self.assertFalse(output.exists())
        with patch("sys.argv", ["probes", "--spec", str(spec_path)]):
            with contextlib.redirect_stdout(io.StringIO()):
                probes_main()
        self.assertEqual(json.loads((output / "prefix_bank.json").read_text())["actual"], 4)
        self.assertEqual(json.loads((output / "status.json").read_text())["status"], "completed")


class PairedStatistics(unittest.TestCase):
    def episode(self, task, success, technical=False):
        return {
            "task_id": task,
            "outcome": {"task_success": success, "technical_failure": technical},
        }

    def test_repeated_episodes_do_not_inflate_task_weight(self):
        left = [self.episode("a", False)] * 10 + [self.episode("b", True)]
        right = [self.episode("a", True)] * 10 + [self.episode("b", False)]
        result = compare(left, right, "task_success", replicates=50)
        self.assertEqual(result["paired_tasks"], 2)
        self.assertEqual(result["mean_difference"], 0)
        self.assertEqual(result, compare(left, right, "task_success", replicates=50))

    def test_unknown_unmatched_and_technical_denominators(self):
        left = [self.episode("a", None), self.episode("b", False, True)]
        right = [self.episode("a", True), self.episode("b", True)]
        all_rows = compare(left, right, "task_success", replicates=10)
        self.assertEqual(all_rows["paired_tasks"], 1)
        self.assertEqual(all_rows["right_only_tasks"], ["a"])
        self.assertEqual(all_rows["left_counts"]["missing_metric"], 1)
        self.assertIsNone(all_rows["bootstrap_percentile_95"])
        filtered = compare(left, right, "task_success", replicates=10, exclude_technical=True)
        self.assertEqual(filtered["left_counts"]["excluded_technical"], 1)
        self.assertEqual(filtered["paired_tasks"], 0)
        self.assertIsNone(filtered["mean_difference"])
