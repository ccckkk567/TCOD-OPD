"""Offline intervention validity checks; no native update or HPC claims."""

import copy
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from opd_pilot.confirmation import FIELDS
from opd_pilot.confirmation import freeze as freeze_protocol
from opd_pilot.confirmation import verify
from opd_pilot.update_probe import (
    alignment,
    compare,
    digest,
    freeze,
    load_bundle,
    plan,
    validate_batch,
)


def batch_fixture():
    return dict(
        purpose="e5_fresh_student_train",
        official_split="train",
        student_revision="student-step25",
        checkpoint_revision="student-step25",
        sampling={"temperature": 1},
        rows=[
            dict(
                sample_id="episode/turn1",
                tokens=[1, 2, 3],
                prompt_length=1,
                response_mask=[1, 1],
                native_fields={"rollout_log_probs": [-1.0, -2.0]},
            )
        ],
    )


class OfflineProbes(unittest.TestCase):
    def test_plan_keeps_source_read_only_without_claiming_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actor = root / "checkpoint/actor"
            actor.mkdir(parents=True)
            for kind in ("model", "optim", "extra_state"):
                with zipfile.ZipFile(actor / f"{kind}_world_size_1_rank_0.pt", "w") as archive:
                    archive.writestr("archive/data.pkl", "structure fixture; not real tensors")
            batch = batch_fixture()
            freeze("batch", batch, root / "batch")
            score = dict(
                batch_digest=digest(batch),
                teacher_revision="teacher",
                tokenizer_revision="tok",
                backend="fixture",
                temperature=1,
                rows=copy.deepcopy(batch["rows"]),
            )
            score["rows"][0].update(teacher_log_probs=[-1, -2], valid_mask=[1, 1])
            freeze("score", score, root / "score", batch)
            spec = dict(
                checkpoint=str(actor.parent),
                world_size=1,
                batch=str(root / "batch"),
                scores={"A": str(root / "score"), "B": str(root / "score")},
                branches_root=str(root / "branches"),
            )
            report = plan(spec)
            self.assertTrue(report["structurally_eligible"])
            self.assertFalse(report["checkpoint"]["recovery_validated"])
            self.assertFalse(report["execution_authorized"])
            self.assertFalse((root / "branches").exists())
            spec["branches_root"] = str(actor / "branches")
            with self.assertRaisesRegex(ValueError, "separate"):
                plan(spec)

    def test_freeze_preserves_payload_and_detects_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "batch"
            batch = batch_fixture()
            freeze("batch", batch, output)
            self.assertEqual(load_bundle(output, "batch"), batch)
            batch["rows"][0]["tokens"][1] = 9
            (output / "payload.json").write_text(json.dumps(batch))
            with self.assertRaisesRegex(ValueError, "integrity"):
                load_bundle(output, "batch")

    def test_alignment_reports_mask_order_and_score_failures_without_mutation(self):
        batch = batch_fixture()
        score = dict(
            batch_digest=digest(batch),
            teacher_revision="teacher",
            tokenizer_revision="tok",
            backend="backend",
            temperature=1,
            rows=copy.deepcopy(batch["rows"]),
        )
        row = score["rows"][0]
        row.update(teacher_log_probs=[-0.5, -0.3], valid_mask=[1, 1])
        self.assertTrue(alignment(batch, score)["aligned"])
        row["tokens"] = [1, 3, 2]
        row["valid_mask"] = [1, 0]
        row["teacher_log_probs"][0] = float("nan")
        report = alignment(batch, score)
        self.assertEqual(
            set(report["differences"]),
            {"rows[0].tokens", "rows[0].valid_mask", "rows[0].nonfinite_score"},
        )
        self.assertEqual(batch["rows"][0]["response_mask"], [1, 1])
        batch["official_split"] = "confirm_holdout"
        with self.assertRaisesRegex(ValueError, "holdout"):
            validate_batch(batch)

    def test_comparison_preserves_missing_and_checks_protocol(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.json"
            path.write_text(json.dumps(dict(protocol_digest="protocol", metrics={"nll": 2})))
            spec = dict(
                results={"C": str(path), "A": None, "B": str(path)},
                protocol_digest="protocol",
                metrics=["nll", "success"],
            )
            report = compare(spec)
            self.assertIsNone(report["metrics"]["nll"]["A-C"])
            self.assertIsNone(report["metrics"]["success"]["B-C"])
            self.assertEqual(report["metrics"]["nll"]["B-C"], 0)
            spec["protocol_digest"] = "different"
            with self.assertRaisesRegex(ValueError, "protocols"):
                compare(spec)

    def test_confirmation_lock_rejects_overlap_and_detects_task_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            holdout, explored = root / "holdout.jsonl", root / "explored.jsonl"
            holdout.write_text('{"task_id":"new-task"}\n')
            explored.write_text('{"task_id":"new-task"}\n')
            protocol = dict.fromkeys(FIELDS, "explicit fixture choice")
            protocol.update(
                holdout_tasks=str(holdout),
                exploration_tasks=str(explored),
                confirmation_results_seen=False,
                budget=dict(tasks=1, repeats=2, max_env_steps=30, max_new_tokens=512),
                seeds=dict(environment=1, candidate=2, continuation=3),
            )
            with self.assertRaisesRegex(ValueError, "overlap"):
                freeze_protocol(protocol, root / "lock")
            explored.write_text('{"task_id":"old-task"}\n')
            freeze_protocol(protocol, root / "lock")
            self.assertEqual(verify(root / "lock")["status"], "protocol_integrity_passed")
            holdout.write_text('{"task_id":"changed-task"}\n')
            with self.assertRaisesRegex(ValueError, "Task list changed"):
                verify(root / "lock")
