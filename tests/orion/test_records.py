"""Check evidence isolation, backpressure, linkage and visible write failures."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_pilot.records import Recorder, audit


class Records(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def turn(self):
        return dict(
            turn_uid="episode-1-turn-1",
            episode_uid="episode-1",
            sampling_student_revision="student-before-update",
            scoring_teacher_revision="teacher-fixed",
            messages=[{"role": "user", "content": "actual input"}],
            tokens=[10, 20],
            prompt_length=1,
            response="look",
            action="look",
            raw_reward=0,
            student_logprobs=[-0.5],
            teacher_logprobs=[-0.2],
        )

    def test_cross_worker_links_and_snapshot(self):
        turn = self.turn()
        with Recorder(self.root, "explorer-1-session-1", capacity=1) as writer:
            writer.record("turn", **turn)
            turn["messages"][0]["content"] = "later mutation"
            writer.record("episode", episode_uid="episode-1", task_id="task-1", outcome={})
        with Recorder(self.root, "trainer-1-session-1", capacity=1) as writer:
            writer.record(
                "update",
                update_uid="update-1",
                update_student_revision="student-after-update",
                consumed_turn_uids=["episode-1-turn-1"],
                mask=[[1]],
                advantage=[[0.3]],
                effective_token_count=1,
            )
        self.assertEqual(audit(self.root)["records"], {"episode": 1, "turn": 1, "update": 1})
        path = self.root / "explorer-1-session-1" / "events.jsonl"
        first = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        self.assertEqual(first["evidence"]["messages"][0]["content"], "actual input")

    def test_no_overwrite_and_no_dropped_records(self):
        with Recorder(self.root, "worker-1", capacity=1) as writer:
            with self.assertRaises(FileExistsError):
                Recorder(self.root, "worker-1")
            for index in range(100):
                writer.record("episode", episode_uid=f"episode-{index}", task_id="task", outcome={})
        self.assertEqual(audit(self.root)["records"]["episode"], 100)

    def test_missing_references_and_incomplete_shard_rejected(self):
        with Recorder(self.root, "worker-1") as writer:
            writer.record("turn", **self.turn())
        with self.assertRaisesRegex(ValueError, "Dangling"):
            audit(self.root)
        (self.root / "worker-1" / "COMPLETE.json").unlink()
        with self.assertRaises(FileNotFoundError):
            audit(self.root)

    def test_disk_failure_surfaces_without_completion_marker(self):
        with patch.object(Path, "open", side_effect=OSError("disk full")):
            writer = Recorder(self.root, "failed-worker", capacity=1)
            writer._thread.join(timeout=2)
            with self.assertRaisesRegex(RuntimeError, "Evidence writer failed"):
                writer.close()
        self.assertFalse((self.root / "failed-worker" / "COMPLETE.json").exists())
