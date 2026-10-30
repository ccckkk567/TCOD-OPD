"""Check infrastructure-only rendering and detection of incomplete rank saves."""

import copy
import tempfile
import unittest
import zipfile
from pathlib import Path

from opd_pilot.checkpoints import inspect
from opd_pilot.configs import REMOTE, STORAGE, differences, render


class ConfigRendering(unittest.TestCase):
    def setUp(self):
        self.template = {
            "algorithm": {"advantage_fn": "multi_turn_opd"},
            "model": {"model_path": "old", "max_response_tokens": 512},
            "cluster": {"gpu_per_node": 8},
            "trainer": {"total_steps": 250},
            "explorer": {
                "auxiliary_models": [{"model_path": "teacher", "tensor_parallel_size": 2}],
                "env_vars": {},
                "rollout_model": {"engine_num": 2},
            },
            "buffer": {
                "total_steps": 250,
                "explorer_input": {"taskset": {"path": "old", "rollout_args": {"temperature": 1}}},
                "trainer_input": {"experience_buffer": {"path": "old"}},
            },
        }
        self.spec = {
            "run_name": "alfworld-s17",
            "student": "s17",
            "teacher": "t08",
            "train_tasks": STORAGE
            + "/outputs/tcod-opd/protocols/alfworld/pilot-v1/train_pilot.jsonl",
        }

    def test_preserves_algorithm_sampling_resources_and_steps(self):
        original = copy.deepcopy(self.template)
        result = render(self.spec, self.template)
        self.assertEqual(self.template, original)
        for key in ("algorithm", "cluster", "trainer"):
            self.assertEqual(result[key], original[key])
        self.assertEqual(result["buffer"]["total_steps"], 250)
        self.assertEqual(result["model"]["model_path"], REMOTE + "/models/Qwen3-1.7B")
        changed = {row["field"] for row in differences(original, result)}
        self.assertFalse(any(key.startswith("algorithm.") for key in changed))

    def test_unique_run_paths_and_reject_holdout_training(self):
        a = render(self.spec, self.template)
        b = render(dict(self.spec, run_name="alfworld-s04", student="s04"), self.template)
        self.assertNotEqual(a["ray_namespace"], b["ray_namespace"])
        self.assertNotEqual(a["buffer"]["trainer_input"], b["buffer"]["trainer_input"])
        with self.assertRaises(ValueError):
            render(
                dict(
                    self.spec,
                    train_tasks=self.spec["train_tasks"].replace("train_pilot", "confirm_holdout"),
                ),
                self.template,
            )


class Checkpoints(unittest.TestCase):
    def test_missing_truncated_and_complete_structures(self):
        with tempfile.TemporaryDirectory() as temp:
            actor = Path(temp) / "actor"
            actor.mkdir()
            self.assertEqual(len(inspect(temp, 2)["missing"]), 6)
            for rank in range(2):
                for kind in ("model", "optim", "extra_state"):
                    path = actor / f"{kind}_world_size_2_rank_{rank}.pt"
                    with zipfile.ZipFile(path, "w") as archive:
                        archive.writestr("archive/data.pkl", b"synthetic structure fixture")
            result = inspect(temp, 2)
            self.assertEqual(result["status"], "structure_complete")
            self.assertFalse(result["recovery_validated"])
            (actor / "optim_world_size_2_rank_1.pt").write_bytes(b"truncated")
            self.assertEqual(inspect(temp, 2)["invalid"], ["optim_world_size_2_rank_1.pt"])
