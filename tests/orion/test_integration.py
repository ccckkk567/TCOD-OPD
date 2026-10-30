"""Exercise production observer call sites with deterministic model/environment doubles."""

import ast
import asyncio
import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List
from unittest.mock import patch

from opd_pilot.evidence import read_records
from opd_pilot.integration import episode_trace, update_trace
from opd_pilot.records import audit

ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = ("alfworld", "webshop", "scienceworld")


class Array(list):
    """Tiny tensor double; CI needs no NumPy, PyTorch or benchmark installation."""

    def tolist(self):
        return copy.deepcopy(list(self))

    def detach(self):
        return self

    def cpu(self):
        return self

    def __getitem__(self, key):
        result = super().__getitem__(key)
        return Array(result) if isinstance(key, slice) else result

    def __sub__(self, other):
        return Array(left - right for left, right in zip(self, other))

    def sum(self):
        value = sum(sum(row) if isinstance(row, list) else row for row in self)
        return NS(item=lambda: value)


def workflow(benchmark, calls, terminal=True, failure=None, invalid=False):
    """Compile the real workflow methods, substituting only external dependencies."""
    path = ROOT / f"trinity/common/workflows/envs/TCOD/{benchmark}/OPD_workflow.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    cls.decorator_list = []
    cls.bases = []
    cls.body = [
        node
        for node in cls.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in ("run_async", "_run_episode", "compute_reward", "format_messages")
    ]
    env = NS(observation="start", steps=0)

    def reset(*args, **kwargs):
        calls.append(("reset", args, kwargs))
        env.observation = "start"
        env.steps = 0
        return "start", {"admissible_commands": [["look", "help"]], "score": 0}

    def step(action):
        calls.append(("step", action))
        if failure == "step":
            raise RuntimeError("environment failed")
        env.steps += 1
        return "next", 0.0, terminal, {"score": 40 if env.steps == 1 else 10}

    def science_reset(env):
        observation, info = env.reset()
        return observation, info, "task"

    def create(*args, **kwargs):
        calls.append(("create", args, kwargs))
        return env

    async def chat(messages, **kwargs):
        calls.append(("student", copy.deepcopy(messages), kwargs))
        if failure == "student":
            raise RuntimeError("student failed")
        return [
            NS(
                tokens=Array([10, 20, 30, 40]),
                prompt_length=2,
                logprobs=Array([-0.8, -0.6]),
                response_text="look",
                teacher_logprobs=None,
                reward=None,
                metrics=None,
                info={},
                eid=NS(uid="temporary", run=0, step=0),
            )
        ]

    async def teacher(**kwargs):
        calls.append(("teacher", kwargs))
        if failure == "teacher":
            raise RuntimeError("teacher failed")
        return Array([-1.0, -0.3, -0.2])

    env.reset, env.step = reset, step
    env.close = lambda: calls.append(("close",))
    env.get_available_actions = lambda: ["look"]
    scope = dict(
        List=List,
        Experience=object,
        episode_trace=episode_trace,
        HISTORY_LENGTH=1,
        parse_action=lambda text: text,
        format_observation=lambda obs: obs,
        _format_history=lambda *args: str(args),
        _extract_task=lambda obs: "task",
        _extract_task_description=lambda obs: "task",
        _format_available_actions=lambda actions: str(actions),
        validate_action=lambda action, available: (not invalid, "invalid action"),
        _get_compact_action_info=lambda env: (["help", "look"], ["object"]),
        _reset_scienceworld_env=science_reset,
        _create_alfworld_env=create,
        _create_scienceworld_env=create,
    )
    for prefix in ("ALFWORLD", "WEBSHOP", "SCIWORLD"):
        scope[prefix + "_TEMPLATE"] = "history={action_history};obs={current_observation}"
        scope[prefix + "_TEMPLATE_NO_HIS"] = "obs={current_observation}"
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec"), scope)
    instance = scope[cls.name]()
    instance.task = NS(raw_task={"task_id": "task-1"}, task_desc="1")
    instance.task_desc, instance.run_id_base = "1", 7
    instance.repeat_times = 2 if benchmark == "webshop" else 1
    instance.rollout_args = {"temperature": 0.7, "logprobs": None}
    instance.temperature, instance.max_env_steps = 0.7, 2
    instance.env, instance.model, instance.teacher_model = (
        env,
        NS(chat_async=chat),
        NS(logprobs_async=teacher),
    )
    return instance


def actor_boundary():
    """Run the actual actor-update with-block without loading Ray/veRL/CUDA locally."""
    path = ROOT / "trinity/trainer/verl_trainer.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    node = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.With)
        and isinstance(node.items[0].context_expr, ast.Call)
        and isinstance(node.items[0].context_expr.func, ast.Name)
        and node.items[0].context_expr.func.id == "update_trace"
    )
    wrapper = ast.parse("def invoke(self, batch, batch_exps):\n    pass\n").body[0]
    wrapper.body = [node, ast.Return(value=ast.Name(id="actor_output", ctx=ast.Load()))]
    module = ast.fix_missing_locations(ast.Module(body=[wrapper], type_ignores=[]))
    scope = {"update_trace": update_trace}
    exec(compile(module, str(path), "exec"), scope)
    return scope["invoke"]


class Integration(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.env = {
            "OPD_RECORD_ROOT": str(self.root),
            "OPD_STUDENT_REVISION": "student-base",
            "OPD_TEACHER_REVISION": "teacher-fixed",
        }

    def run_workflow(self, benchmark, enabled=True, **kwargs):
        calls = []
        instance = workflow(benchmark, calls, **kwargs)
        with patch.dict(os.environ, self.env if enabled else {"OPD_RECORD_ROOT": ""}):
            result = asyncio.run(instance.run_async())
        return calls, result

    def test_all_workflows_preserve_calls_outputs_and_reward_rules(self):
        for benchmark in BENCHMARKS:
            for terminal in (True, False):
                with self.subTest(benchmark=benchmark, terminal=terminal):
                    before_calls, before = self.run_workflow(benchmark, False, terminal=terminal)
                    after_calls, after = self.run_workflow(benchmark, terminal=terminal)
                    self.assertEqual(before_calls, after_calls)
                    self.assertEqual(len(before), len(after))
                    expected_reward = (
                        (1.0 if terminal else 0.0)
                        if benchmark == "alfworld"
                        else (0.4 if benchmark == "scienceworld" else 0.0)
                    )
                    for a, b in zip(before, after):
                        self.assertEqual(a.reward, expected_reward)
                        self.assertEqual(a.reward, b.reward)
                        self.assertEqual(a.metrics, b.metrics)
                        self.assertEqual(vars(a.eid), vars(b.eid))
                        self.assertEqual(a.info, {})
                        self.assertEqual(set(b.info), {"opd_turn_uid"})
                        for field in ("tokens", "logprobs", "teacher_logprobs"):
                            self.assertEqual(getattr(a, field).tolist(), getattr(b, field).tolist())
        records = read_records(self.root)
        for turn in records["turn"]:
            self.assertEqual(turn["messages"][-1]["role"], "user")
            self.assertEqual(turn["teacher_logprobs"], [-0.3, -0.2])
            self.assertTrue(turn["teacher_scoring_completed"])
            self.assertEqual(turn["sampling_kwargs"], {"temperature": 0.7, "logprobs": 0, "n": 1})
            self.assertEqual(turn["scoring_temperature"], 0.7)
        self.assertTrue(all(ep["outcome"]["task_success"] is None for ep in records["episode"]))
        science_horizon = next(
            ep["outcome"]
            for ep in records["episode"]
            if ep["benchmark"] == "scienceworld" and not ep["outcome"]["episode_done"]
        )
        self.assertEqual(science_horizon["score_final"], 10)
        self.assertEqual(science_horizon["score_best"], 40)
        self.assertEqual(science_horizon["tcod_final_reward"], 0.4)

    def test_webshop_invalid_actions_do_not_add_environment_calls(self):
        before, _ = self.run_workflow("webshop", False, invalid=True)
        after, _ = self.run_workflow("webshop", invalid=True)
        self.assertEqual(before, after)
        self.assertFalse(any(call[0] == "step" for call in after))
        records = read_records(self.root)
        self.assertEqual(len(records["episode"]), 2)
        self.assertEqual(len({row["turn_uid"] for row in records["turn"]}), 4)
        self.assertTrue(all(row["environment_step_executed"] is False for row in records["turn"]))

    def test_workflow_failures_propagate_and_keep_partial_evidence(self):
        for benchmark in BENCHMARKS:
            for failure in ("student", "step", "teacher"):
                with self.subTest(benchmark=benchmark, failure=failure):
                    before, after = [], []
                    with patch.dict(os.environ, {"OPD_RECORD_ROOT": ""}):
                        with self.assertRaisesRegex(RuntimeError, "failed"):
                            asyncio.run(workflow(benchmark, before, failure=failure).run_async())
                    with patch.dict(os.environ, self.env):
                        with self.assertRaisesRegex(RuntimeError, "failed"):
                            asyncio.run(workflow(benchmark, after, failure=failure).run_async())
                    self.assertEqual(before, after)
        records = read_records(self.root)
        self.assertEqual(len(records["episode"]), 9)
        self.assertEqual(len(records["turn"]), 6)
        self.assertTrue(all(ep["outcome"]["technical_failure"] for ep in records["episode"]))
        self.assertTrue(all(not row["teacher_scoring_completed"] for row in records["turn"]))

    def test_actor_boundary_links_reordered_consumed_rows_and_preserves_batch(self):
        _, experiences = self.run_workflow("alfworld", terminal=False)
        for index, exp in enumerate(experiences):
            # Runner assigns final IDs and model versions after workflow completion.
            exp.eid.uid = f"batch/task/{index}"
            exp.info["model_version"] = index + 2
        batch = NS(
            non_tensor_batch={"unique_ids": Array(["batch/task/1", "batch/task/0"])},
            batch={
                "response_mask": Array([[1, 0], [1, 1]]),
                "advantages": Array([[0.2, 0], [0.3, 0.4]]),
            },
        )
        original = copy.deepcopy(batch)
        calls = []
        sentinel = object()

        def update(actual):
            self.assertIs(actual, batch)
            calls.append("update")
            return sentinel

        trainer = NS(global_steps=5, _update_actor=update)
        with patch.dict(os.environ, self.env):
            self.assertIs(actor_boundary()(trainer, batch, experiences), sentinel)
        self.assertEqual(calls, ["update"])
        for key in batch.batch:
            self.assertEqual(batch.batch[key].tolist(), original.batch[key].tolist())
        records = read_records(self.root)
        event = records["update"][0]
        self.assertEqual(
            event["consumed_turn_uids"], [exp.info["opd_turn_uid"] for exp in reversed(experiences)]
        )
        self.assertEqual(event["sampling_model_versions"], [3, 2])
        self.assertEqual(event["advantage"], [[0.2, 0], [0.3, 0.4]])
        self.assertEqual(event["effective_token_count"], 3)
        self.assertEqual(audit(self.root)["unresolved_update_attempts"], [])

        def failed_update(actual):
            calls.append("failed update")
            raise RuntimeError("actor failed")

        trainer._update_actor = failed_update
        with patch.dict(os.environ, self.env):
            with self.assertRaisesRegex(RuntimeError, "actor failed"):
                actor_boundary()(trainer, batch, experiences)
        records = read_records(self.root)
        self.assertEqual(len(records["update"]), 1)
        self.assertEqual(len(records["update_attempt"]), 2)
        self.assertEqual(records["update_failure"][0]["status"], "failed_or_partially_applied")

    def test_disabled_recording_needs_no_metadata_and_makes_no_files(self):
        self.run_workflow("alfworld", enabled=False)
        with patch.dict(os.environ, {"OPD_RECORD_ROOT": ""}):
            with update_trace(None, None, None) as trace:
                self.assertIsNone(trace)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_enabled_recording_requires_explicit_revisions(self):
        with patch.dict(os.environ, {**self.env, "OPD_TEACHER_REVISION": ""}):
            with self.assertRaisesRegex(ValueError, "OPD_TEACHER_REVISION"):
                episode_trace(NS(), "alfworld")

    def test_record_write_failure_stops_the_workflow(self):
        with patch("opd_pilot.integration.Recorder.record", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                self.run_workflow("scienceworld")
