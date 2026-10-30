"""Verify approved step/schedule changes while retaining the upstream algorithm."""

import ast
import copy
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from opd_pilot.pilot_config import configure


class PilotConfig(unittest.TestCase):
    def test_schedule_horizon_uses_actual_sync_expression(self):
        path = Path(__file__).resolve().parents[2] / "trinity/common/verl_config.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assignment = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                ast.unparse(target) == "self.trainer.total_training_steps"
                for target in node.targets
            )
        )
        for schedule, target, expected in (
            (100, 2, 100),
            (100, 4, 100),
            (100, 50, 100),
            (None, 250, 250),
        ):
            instance = NS(trainer=NS())
            config = NS(trainer=NS(schedule_steps=schedule, total_steps=target))
            exec(
                compile(ast.Module(body=[assignment], type_ignores=[]), str(path), "exec"),
                dict(self=instance, config=config, sys=sys),
            )
            self.assertEqual(instance.trainer.total_training_steps, expected)

    def test_phases_change_only_approved_fields(self):
        template = dict(
            project="upstream",
            name="example",
            checkpoint_root_dir="old",
            ray_namespace="old",
            model=dict(model_path="base"),
            explorer=dict(auxiliary_models=[dict(model_path="teacher")]),
            buffer=dict(
                explorer_input=dict(taskset=dict(path="old")),
                trainer_input=dict(experience_buffer=dict(path="old")),
                total_steps=250,
            ),
            trainer=dict(total_steps=250, save_interval=250),
            cluster=dict(node_num=1, gpu_per_node=8),
            algorithm=dict(algorithm_type="on_policy_distill", optimizer=dict(lr=1e-6)),
        )
        original = copy.deepcopy(template)
        for phase, target in (("smoke-first", 2), ("smoke-resume", 4), ("pilot", 50)):
            spec = dict(
                run_name="test",
                student="s17",
                teacher="t08",
                phase=phase,
                train_tasks="/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/outputs/tcod-opd/protocols/alfworld/pilot-v1/train_pilot.jsonl",
            )
            config = configure(spec, template)
            self.assertEqual(config["trainer"]["schedule_steps"], 100)
            self.assertEqual(config["trainer"]["total_steps"], target)
            self.assertEqual(config["algorithm"], original["algorithm"])
            self.assertEqual(config["cluster"], original["cluster"])
            self.assertEqual(config["continue_from_checkpoint"], phase == "smoke-resume")
        self.assertEqual(template, original)

    def test_explicit_save_steps_and_original_interval(self):
        path = Path(__file__).resolve().parents[2] / "trinity/trainer/trainer.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "need_save"
        )
        scope = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), scope)
        trainer = NS(config=NS(trainer=NS(save_steps=[10, 25, 50])), save_interval=0)
        for step in (1, 10, 24, 25, 49, 50):
            trainer.train_step_num = step
            self.assertEqual(scope["need_save"](trainer), step in (10, 25, 50))
        trainer.config.trainer.save_steps = []
        trainer.save_interval = 5
        trainer.train_step_num = 15
        self.assertTrue(scope["need_save"](trainer))
