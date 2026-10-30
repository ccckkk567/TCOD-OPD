"""Verify diagnostic prompts, failure denominators, and replay before interventions."""

import asyncio
import os
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from test_integration import workflow

from opd_pilot.continuation import ReplayMismatch, branch, restore
from opd_pilot.environment import Episode
from opd_pilot.evaluate import rollout


class DiagnosticEpisodes(unittest.TestCase):
    def setup_episode(self, benchmark):
        calls = []
        original = workflow(benchmark, calls, terminal=False)
        original.repeat_times = 1
        scope = original.run_async.__func__.__globals__
        utilities = NS(**scope)
        utilities._create_webshop_env = lambda: original.env
        return original, calls, utilities

    def test_same_prompts_actions_and_history_as_each_original_workflow(self):
        for benchmark in ("alfworld", "webshop", "scienceworld"):
            original, calls, utilities = self.setup_episode(benchmark)
            with patch.dict(os.environ, {"OPD_RECORD_ROOT": ""}):
                asyncio.run(original.run_async())
            expected = [call[1] for call in calls if call[0] == "student"]
            expected_actions = [call[1] for call in calls if call[0] == "step"]
            env = Episode(benchmark, "1", 2, utilities=utilities)
            messages, actions = [], []
            try:
                for step in range(2):
                    messages.append(env.prompt())
                    actions.append(env.apply("look")["action"])
                self.assertEqual(messages, expected)
                self.assertEqual(actions, expected_actions)
            finally:
                env.close()

    def test_replay_checks_state_before_allowing_candidate(self):
        _, _, utilities = self.setup_episode("alfworld")
        env = Episode("alfworld", "1", 2, utilities=utilities)
        messages = env.prompt()
        transition = env.apply("look")
        target = dict(
            messages=env.prompt(), observation_before=env.observation, info_before=env.info
        )
        env.close()
        history = [dict(messages=messages, response="look", **transition)]
        restored = Episode("alfworld", "1", 2, utilities=utilities)
        try:
            self.assertEqual(restore(restored, history, target), target["messages"])
        finally:
            restored.close()
        target["observation_before"] = "different state"
        restored = Episode("alfworld", "1", 2, utilities=utilities)
        try:
            with self.assertRaises(ReplayMismatch):
                restore(restored, history, target)
        finally:
            restored.close()

    def test_generation_failure_is_retained_and_environment_closed(self):
        env = Mock()
        env.prompt.return_value = [{"role": "user", "content": "task"}]
        backend = Mock()
        backend.generate.side_effect = RuntimeError("GPU error")
        row = rollout(
            dict(benchmark="alfworld", max_env_steps=2, seed=42),
            dict(task_id="task", task_desc="game"),
            0,
            backend,
            factory=lambda *args: env,
        )
        self.assertTrue(row["outcome"]["technical_failure"])
        self.assertIsNone(row["outcome"]["task_success"])
        env.close.assert_called_once()
        env.apply.assert_not_called()

    def test_branch_keeps_full_candidate_in_executor_history(self):
        _, _, utilities = self.setup_episode("alfworld")

        def factory(benchmark, task, limit):
            return Episode(benchmark, task, limit, utilities=utilities)

        initial = factory("alfworld", "1", 2)
        target = dict(
            messages=initial.prompt(),
            observation_before=initial.observation,
            info_before=initial.info,
        )
        initial.close()
        candidate = dict(
            candidate_uid="teacher-prefix-1",
            prefix_uid="prefix-1",
            task_id="task-1",
            owner="teacher",
            response_text="reasoning plus proposed action",
        )
        backend = Mock()
        backend.encode_messages.return_value = [1, 2]
        backend.generate.return_value = dict(response_text="look", response_tokens=[3])
        result = branch(
            dict(benchmark="alfworld", executor="student", max_env_steps=2, seed=42),
            candidate,
            [],
            target,
            dict(task_desc="1"),
            0,
            backend,
            factory,
        )
        self.assertTrue(result["replay_validated"])
        self.assertFalse(result["outcome"]["technical_failure"])
        messages = backend.encode_messages.call_args.args[0]
        self.assertIn(dict(role="assistant", content=candidate["response_text"]), messages)
        self.assertEqual(len(result["turns"]), 2)
