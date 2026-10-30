"""Check that the pilot gate rejects unrelated runs and unsupported schedules."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import yaml

from orion.scripts.pilot_recovery_gate import check, expected_lr


def resolved():
    return dict(
        cluster=dict(trainer_gpu_num=1, shared_inference_gpu=True),
        checkpoint_job_dir="/own/checkpoints/smoke",
        algorithm=dict(optimizer=dict(lr=1e-6)),
        explorer=dict(auxiliary_models=[dict(model_path="/own/models/teacher")]),
        trainer=dict(
            trainer_config=dict(
                actor_rollout_ref=dict(
                    actor=dict(
                        optim=dict(
                            lr=1e-6,
                            lr_scheduler_type="constant",
                            lr_warmup_steps_ratio=0,
                            lr_warmup_steps=-1,
                        )
                    )
                )
            )
        ),
    )


class PilotGateTests(unittest.TestCase):
    def test_lr_expectation_does_not_accept_unknown_schedule(self):
        cfg = resolved()
        self.assertEqual(expected_lr(cfg), 1e-6)
        cfg["trainer"]["trainer_config"]["actor_rollout_ref"]["actor"]["optim"][
            "lr_scheduler_type"
        ] = "cosine"
        with self.assertRaises(ValueError):
            expected_lr(cfg)

    def test_expected_job_and_real_recovery_checker_required(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for phase, start, target in (("smoke-first", 0, 2), ("smoke-resume", 2, 4)):
                p = root / phase
                p.mkdir()
                (p / "COMPLETE.json").write_text(
                    json.dumps(
                        dict(
                            status="completed",
                            start_step=start,
                            final_step=target,
                            actual_updates=2,
                        )
                    )
                )
                (p / "resolved-config.yaml").write_text(yaml.safe_dump(resolved()))
            (root / "smoke-resume/run-metadata.json").write_text(json.dumps(dict(job_id="182356")))
            with patch("orion.scripts.pilot_recovery_gate.recovery") as recovery, patch(
                "orion.scripts.pilot_recovery_gate.model_manifest",
                return_value=dict(revision="teacher"),
            ):
                with self.assertRaisesRegex(ValueError, "different Slurm"):
                    check(root, "999", root / "report.json")
                recovery.assert_not_called()
                recovery.side_effect = ValueError("optimizer counters reset")
                with self.assertRaisesRegex(ValueError, "optimizer counters reset"):
                    check(root, "182356", root / "report.json")
                self.assertFalse((root / "report.json").exists())
                recovery.side_effect = None
                recovery.return_value = dict(status="recovery_evidence_consistent")
                check(root, "182356", root / "report.json")
                self.assertEqual(recovery.call_args.args[0]["expected_final_lrs"], [1e-6])
                self.assertTrue((root / "report.json").exists())
