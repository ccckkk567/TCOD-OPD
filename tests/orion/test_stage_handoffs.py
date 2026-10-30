"""Verify evidence handoffs and reject false completion without GPU dependencies."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_pilot.evaluate import write_records
from opd_pilot.evidence import publish, read_records
from opd_pilot.prepare import prepare_alfworld
from opd_pilot.probes import select
from opd_pilot.records import Recorder
from opd_pilot.task_protocol import split
from opd_pilot.train import completion_evidence


class StageHandoffs(unittest.TestCase):
    def test_resumed_updates_can_reference_prior_segment_without_recounting_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            before, after = root / "before", root / "after"
            with Recorder(before, "explorer") as writer:
                writer.record("episode", episode_uid="episode", task_id="task", outcome={})
                writer.record(
                    "turn",
                    turn_uid="turn",
                    episode_uid="episode",
                    sampling_student_revision="student",
                    scoring_teacher_revision="teacher",
                    messages=[],
                    tokens=[1, 2],
                    prompt_length=1,
                    response="look",
                    action="look",
                    raw_reward=0,
                    student_logprobs=[-1],
                    teacher_logprobs=[-1],
                )
            with Recorder(after, "trainer") as writer:
                writer.record(
                    "update",
                    update_uid="update3",
                    update_student_revision="student",
                    consumed_turn_uids=["turn"],
                    mask=[[1]],
                    advantage=[[0]],
                    effective_token_count=1,
                    trainer_step_after_if_completed=3,
                )
            with self.assertRaisesRegex(ValueError, "Dangling"):
                read_records(after)
            records = read_records(after, reference_roots=[before])
            self.assertEqual(len(records["update"]), 1)
            self.assertEqual(records["turn"], [])

    def test_evaluation_records_enter_dev_probes_without_invented_teacher_scores(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(4):
                game = root / "data/json_2.1.1/train" / f"pick-{index}/trial/game.tw-pddl"
                game.parent.mkdir(parents=True)
                game.write_text("test fixture")
            protocol = root / "protocol"
            counts = dict(train_pilot=1, analysis_dev=1, confirm_holdout=1, smoke=1)
            prepare_alfworld(root / "data", protocol, counts=counts)
            task = json.loads((protocol / "analysis_dev.jsonl").read_text())
            turn = dict(
                prefix_tokens=[1, 2],
                response_tokens=[3],
                messages=[dict(role="user", content="actual prompt")],
                response_text="look",
                observation_before="room",
                info_before={},
                observation_after="room",
                info_after={},
                action="look",
                raw_reward=0,
                episode_done=False,
            )
            episodes = [
                dict(
                    task_id=task["task_id"],
                    turns=[turn],
                    outcome=dict(task_success=None, technical_failure=False),
                )
            ]
            output = root / "evaluation"
            publish(
                output,
                {"episodes.json": episodes},
                extra_writer=lambda staging: write_records(
                    staging, episodes, {"revision": "student"}, "alfworld"
                ),
            )
            records = read_records(output / "records")
            self.assertIsNone(records["turn"][0]["teacher_logprobs"])
            bank = select(
                dict(
                    records_root=str(output / "records"),
                    protocol_root=str(protocol),
                    split="analysis_dev",
                    student_revision="student",
                    teacher_revision=None,
                    max_prefixes=2,
                    max_per_task=1,
                    seed=42,
                )
            )
            self.assertEqual(bank["prefixes"][0]["prefix_tokens"], [1, 2])
            self.assertEqual(bank["prefixes"][0]["messages"], turn["messages"])

    def test_failed_extra_writer_does_not_publish_artifact(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "result"

            def fail(staging):
                raise OSError("disk full")

            with self.assertRaises(OSError):
                publish(output, {"status.json": {"status": "completed"}}, extra_writer=fail)
            self.assertFalse(output.exists())

    def test_train_completion_requires_real_contiguous_updates_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "training-status.json").write_text(
                json.dumps(dict(status="completed", completed_step=4))
            )
            records = {"update": [dict(trainer_step_after_if_completed=step) for step in (3, 4)]}
            with patch("opd_pilot.train.read_records", return_value=records), patch(
                "opd_pilot.train.inspect", return_value={"status": "incomplete"}
            ):
                with self.assertRaisesRegex(RuntimeError, "checkpoint"):
                    completion_evidence(root, root, root, 4, 2, 2)
            self.assertFalse((root / "COMPLETE.json").exists())
            records["update"].pop()
            with patch("opd_pilot.train.read_records", return_value=records):
                with self.assertRaisesRegex(RuntimeError, "update"):
                    completion_evidence(root, root, root, 4, 2, 2)
            records["update"].append(dict(trainer_step_after_if_completed=4))
            with patch("opd_pilot.train.read_records", return_value=records), patch(
                "opd_pilot.train.inspect", return_value={"status": "structure_complete"}
            ):
                self.assertEqual(
                    completion_evidence(root, root, root, 4, 2, 2)["actual_updates"], 2
                )

    def test_generic_protocol_is_disjoint_and_rejects_test_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            rows = [
                dict(task_id=str(i), task_desc=str(i), official_split="train", stratum=str(i % 2))
                for i in range(8)
            ]
            path.write_text("\n".join(map(json.dumps, rows)))
            spec = dict(
                tasks=str(path),
                seed=42,
                counts=dict(train_pilot=2, analysis_dev=2, confirm_holdout=2, smoke=2),
            )
            result = split(spec)
            self.assertEqual(len({r["task_id"] for group in result.values() for r in group}), 8)
            rows[0]["official_split"] = "test"
            path.write_text("\n".join(map(json.dumps, rows)))
            with self.assertRaisesRegex(ValueError, "training pool"):
                split(spec)
