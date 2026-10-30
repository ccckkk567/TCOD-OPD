"""Check three-turn history and cleanup using original helpers and a simulator double."""

import runpy
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from orion.scripts.environment_interface_smoke import run_episode


class EnvironmentInterfaceSmoke(unittest.TestCase):
    def helpers(self, benchmark):
        repo = Path(__file__).resolve().parents[2]
        helpers = SimpleNamespace(
            **runpy.run_path(
                str(repo / "trinity/common/workflows/envs/TCOD" / benchmark / "utils.py")
            )
        )
        env = Mock()
        observation = "A room. Your task is to: inspect the room"
        info = {"admissible_commands": ["look"], "score": 0}
        env.reset.return_value = observation, info
        env.step.return_value = observation, 0, False, info
        env.get_task_description.return_value = "inspect the room"
        env.get_possible_actions.return_value = ["look around"]
        env.get_possible_objects.return_value = ["room"]
        setattr(helpers, f"_create_{benchmark}_env", Mock(return_value=env))
        return helpers, env

    def test_original_templates_parser_and_history_for_both_benchmarks(self):
        for benchmark in ("alfworld", "scienceworld"):
            with self.subTest(benchmark=benchmark):
                helpers, env = self.helpers(benchmark)
                result = run_episode(benchmark, "task", helpers)
                turns = result["turns"]
                self.assertEqual(len(turns), 3)
                self.assertEqual(turns[2]["messages"][1]["content"], turns[0]["response"])
                self.assertIn("[Observation 1:", turns[2]["messages"][-1]["content"])
                expected = "look" if benchmark == "alfworld" else "look around"
                self.assertEqual([call.args[0] for call in env.step.call_args_list], [expected] * 3)
                env.close.assert_called_once()

    def test_step_failure_propagates_and_closes_environment(self):
        helpers, env = self.helpers("alfworld")
        env.step.side_effect = RuntimeError("simulator failed")
        with self.assertRaisesRegex(RuntimeError, "simulator failed"):
            run_episode("alfworld", "task", helpers)
        env.close.assert_called_once()
