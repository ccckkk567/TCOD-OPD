"""Resource accounting and shared placement checks without a GPU installation."""

import ast
import copy
import json
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from trinity.common.shared_inference import create_shared_inference, validate_shared_inference

ROOT = Path(__file__).resolve().parents[2]


def config(shared=True):
    model = dict(engine_num=1, tensor_parallel_size=1, enforce_eager=True, engine_type="vllm")
    return NS(
        mode="both",
        ray_namespace="test",
        cluster=NS(node_num=1, gpu_per_node=2 if shared else 3, shared_inference_gpu=shared),
        explorer=NS(
            name="explorer",
            rollout_model=NS(**model, gpu_memory_utilization=0.25),
            auxiliary_models=[NS(**model, gpu_memory_utilization=0.55)],
        ),
        trainer=NS(ulysses_sequence_parallel_size=1),
        algorithm=NS(algorithm_type="on_policy_distill"),
    )


def allocate(cfg):
    source = ROOT / "trinity/common/config_validator.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_set_gpu_allocation_info"
    )
    namespace = dict(Config=object)
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    namespace[method.name](None, cfg)


class SharedInferenceTests(unittest.TestCase):
    def test_physical_gpu_accounting_and_default_layout(self):
        cfg = config()
        allocate(cfg)
        self.assertEqual(cfg.cluster.explorer_gpu_num, 1)
        self.assertEqual(cfg.cluster.trainer_gpu_num, 1)
        self.assertEqual(cfg.cluster.trainer_gpu_num_per_node, 1)
        self.assertEqual(cfg.cluster.rollout_gpu_num, 1)  # student NCCL world size
        cfg = config(shared=False)
        allocate(cfg)
        self.assertEqual(cfg.cluster.explorer_gpu_num, 2)
        self.assertEqual(cfg.cluster.trainer_gpu_num, 1)

    def test_reject_unsupported_layout_and_excess_memory(self):
        for field, value in (
            ("engine_num", 2),
            ("tensor_parallel_size", 2),
            ("gpu_memory_utilization", 0.9),
        ):
            cfg = config()
            setattr(cfg.explorer.rollout_model, field, value)
            with self.assertRaises(ValueError):
                validate_shared_inference(cfg)
        cfg = config()
        cfg.trainer.ulysses_sequence_parallel_size = 2
        with self.assertRaises(ValueError):
            validate_shared_inference(cfg)

    def test_actors_share_one_bundle_with_initialization_barriers(self):
        events, options = [], []
        pg = Mock()
        pg.ready.return_value = "ready"
        ray = Mock()
        ray.get_runtime_context.return_value.gcs_address = "10.0.0.1:12345"

        def get(ref, **kwargs):
            events.append("get")
            return None if ref == "ready" else dict(gpu_ids=[0], name="NVIDIA H200")

        ray.get.side_effect = get

        def remote_options(**kwargs):
            events.append("create")
            options.append(kwargs)
            return Mock()

        ray.remote.return_value.options.side_effect = remote_options
        placement = Mock(return_value=pg)
        modules = {
            "ray": ray,
            "ray.util.placement_group": NS(placement_group=placement),
            "ray.util.scheduling_strategies": NS(PlacementGroupSchedulingStrategy=NS),
        }
        with patch.dict(sys.modules, modules):
            students, teachers = create_shared_inference(config(), object)
        self.assertEqual(events, ["get", "create", "get", "create", "get"])
        placement.assert_called_once_with([{"GPU": 1}], strategy="PACK")
        self.assertEqual((len(students), len(teachers[0])), (1, 1))
        for option in options:
            self.assertEqual(option["num_gpus"], 0.5)
            self.assertEqual(option["runtime_env"]["env_vars"]["RAY_ADDRESS"], "10.0.0.1:12345")
            self.assertIs(option["scheduling_strategy"].placement_group, pg)
            self.assertEqual(option["scheduling_strategy"].placement_group_bundle_index, 0)

    def test_generated_config_preserves_algorithm_and_batch(self):
        import yaml
        from opd_pilot.pilot_config import configure

        template = yaml.safe_load((ROOT / "TCOD_examples/alfworld/opd.yaml").read_text())
        original = copy.deepcopy(template)
        for phase in ("smoke-first", "smoke-resume"):
            spec = json.loads((ROOT / f"orion/specs/alfworld-s17-two-gpu-{phase}.json").read_text())
            result = configure(spec, template)
            self.assertEqual(result["algorithm"], original["algorithm"])
            for field in ("batch_size", "train_batch_size"):
                self.assertEqual(result["buffer"][field], original["buffer"][field])
            self.assertEqual(
                result["buffer"]["explorer_input"]["taskset"]["rollout_args"],
                original["buffer"]["explorer_input"]["taskset"]["rollout_args"],
            )
            self.assertEqual(result["synchronizer"], original["synchronizer"])
            self.assertEqual(result["cluster"]["gpu_per_node"], 2)
            self.assertEqual(result["trainer"]["ulysses_sequence_parallel_size"], 1)
        self.assertEqual(template, original)


if __name__ == "__main__":
    unittest.main()
