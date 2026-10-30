"""Exercise the production training loop's completion and failure boundaries."""

import ast
import asyncio
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

from opd_pilot.lifecycle import TrainingIncomplete


def training_loop():
    path = Path(__file__).resolve().parents[2] / "trinity/trainer/trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Trainer")
    method = next(n for n in cls.body if isinstance(n, ast.AsyncFunctionDef) and n.name == "train")
    namespace = dict(
        asyncio=asyncio,
        traceback=traceback,
        RunningStatus=NS(STOPPED="stopped"),
        stop_requested=Mock(return_value=False),
        record_completion=Mock(),
        TrainingIncomplete=TrainingIncomplete,
    )
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    return namespace


class Lifecycle(unittest.TestCase):
    def trainer(self):
        trainer = NS(
            train_step_num=0,
            total_steps=2,
            logger=Mock(),
            monitor=Mock(),
            save_hf_checkpoint="never",
            save_checkpoint=Mock(),
            synchronizer=NS(set_trainer_status=NS(remote=AsyncMock())),
            config=NS(trainer=NS(name="trainer", enable_preview=False)),
            need_sync=AsyncMock(return_value=False),
            need_save=Mock(return_value=False),
            _sample_data=AsyncMock(return_value=(["sample"], {}, [])),
        )

        async def update(exps):
            trainer.train_step_num += 1
            return {}

        trainer.train_step = update
        return trainer

    def run_loop(self, scope, trainer):
        # Doubles never await actual IO; use the normal event loop on CI/Windows.
        return asyncio.run(scope["train"](trainer))

    def test_success_reaches_target_and_preserves_normal_close_owner(self):
        scope, trainer = training_loop(), self.trainer()
        self.assertEqual(self.run_loop(scope, trainer), "trainer")
        scope["record_completion"].assert_called_once_with("completed", 2, 2, True, None)
        trainer.monitor.close.assert_not_called()

    def test_update_failure_is_not_a_completed_step(self):
        scope, trainer = training_loop(), self.trainer()
        failure = ValueError("update failed")

        async def update(exps):
            trainer.train_step_num += 1  # veRL increments its counter before updating.
            raise failure

        trainer.train_step = update
        trainer.save_checkpoint.side_effect = OSError("save also failed")
        with self.assertRaises(ValueError) as raised:
            self.run_loop(scope, trainer)
        self.assertIs(raised.exception, failure)
        scope["record_completion"].assert_called_once_with("failed", 0, 2, False, failure)
        trainer.monitor.close.assert_called_once()

    def test_stop_during_sampling_cancels_wait_then_saves(self):
        scope, trainer = training_loop(), self.trainer()
        scope["stop_requested"].side_effect = [False, True]
        with self.assertRaises(TrainingIncomplete):
            self.run_loop(scope, trainer)
        self.assertEqual(trainer.train_step_num, 0)
        trainer.save_checkpoint.assert_called_once_with(block_until_saved=True, save_as_hf=False)
        scope["record_completion"].assert_called_once_with("partial", 0, 2, True, None)

    def test_final_save_failure_propagates(self):
        scope, trainer = training_loop(), self.trainer()
        trainer.save_checkpoint.side_effect = OSError("disk full")
        with self.assertRaisesRegex(OSError, "disk full"):
            self.run_loop(scope, trainer)
        self.assertEqual(scope["record_completion"].call_args.args[:4], ("failed", 2, 2, False))
