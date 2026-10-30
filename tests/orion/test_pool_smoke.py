"""Check common-pool provenance and reject reset state in recovery evidence."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from opd_pilot.pool import combine
from opd_pilot.smoke import recovery


class PoolSmoke(unittest.TestCase):
    def test_common_pool_preserves_source_state_ids_and_rejects_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            roots = [Path(directory) / name for name in ("student-small", "student-large")]
            for root in roots:
                root.mkdir()
                (root / "spec.json").write_text(
                    json.dumps(dict(split="analysis_dev", records_root=str(root / "records")))
                )
                (root / "prefix_bank.json").write_text(
                    json.dumps(dict(prefixes=[dict(prefix_uid="prefix-1", task_id="same-task")]))
                )
            spec = dict(banks=list(map(str, roots)), max_prefixes=2, seed=42)
            rows = combine(spec)["prefixes"]
            self.assertEqual(len({row["prefix_uid"] for row in rows}), 2)
            self.assertEqual({row["source_prefix_uid"] for row in rows}, {"prefix-1"})
            self.assertEqual(len({row["records_root"] for row in rows}), 2)
            (roots[1] / "spec.json").write_text(json.dumps(dict(split="confirm_holdout")))
            with self.assertRaisesRegex(ValueError, "confirm"):
                combine(spec)

    def test_recovery_requires_continuous_updates_optimizer_and_schedule(self):
        before = dict(
            update=[dict(update_uid="first", trainer_step_after_if_completed=2)],
            turn=[dict(scoring_teacher_revision="teacher")],
        )
        after = dict(
            update=[
                dict(update_uid=f"resume-{step}", trainer_step_after_if_completed=step)
                for step in (3, 4)
            ],
            turn=[dict(scoring_teacher_revision="teacher")],
        )
        spec = dict(
            before_records="before",
            after_records="after",
            resume_step=2,
            target_step=4,
            teacher_revision="teacher",
            world_size=1,
            before_checkpoint="before",
            after_checkpoint="after",
            expected_final_lrs=[0.000001],
        )

        def load(path, **kwargs):
            step = 2 if str(path).startswith("before") else 4
            if path.name.startswith("extra_state"):
                return dict(rng={"state": 1}, lr_scheduler=dict(last_epoch=step))
            return dict(state={0: dict(step=step)}, param_groups=[dict(lr=0.000001)])

        with patch.dict("sys.modules", {"torch": NS(load=load)}), patch(
            "opd_pilot.smoke.read_records", side_effect=[before, after]
        ):
            self.assertEqual(recovery(spec)["status"], "recovery_evidence_consistent")
        spec["expected_final_lrs"] = [0.01]
        with patch.dict("sys.modules", {"torch": NS(load=load)}), patch(
            "opd_pilot.smoke.read_records", side_effect=[before, after]
        ):
            with self.assertRaisesRegex(ValueError, "learning rate"):
                recovery(spec)
