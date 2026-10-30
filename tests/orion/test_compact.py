"""Round-trip actual evidence through numeric-array and message storage."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

from opd_pilot.compact import pack, unpack
from opd_pilot.evidence import read_records
from opd_pilot.records import Recorder


@unittest.skipUnless(
    importlib.util.find_spec("numpy"), "NumPy archive test runs in the analysis environment"
)
class Compact(unittest.TestCase):
    def test_roundtrip_preserves_evidence_and_reuses_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with Recorder(root / "source", "worker") as writer:
                writer.record("episode", episode_uid="episode", task_id="task", outcome={})
                for index in range(2):
                    writer.record(
                        "turn",
                        turn_uid=f"turn-{index}",
                        episode_uid="episode",
                        sampling_student_revision="student",
                        scoring_teacher_revision="teacher",
                        messages=[dict(role="user", content="shared prompt")],
                        tokens=[1, 2, 3],
                        prompt_length=2,
                        response="action",
                        action="action",
                        raw_reward=0,
                        student_logprobs=[-0.1234567890123456],
                        teacher_logprobs=None,
                    )
            result = pack(root / "source", root / "archive")
            self.assertEqual(result["unique_messages"], 1)
            unpack(root / "archive", root / "restored")
            self.assertEqual(read_records(root / "source"), read_records(root / "restored"))
