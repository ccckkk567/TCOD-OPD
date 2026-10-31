"""Exercise real workflow bodies against frozen trajectories, without GPU imports."""

import ast
import asyncio
import copy
import runpy
import subprocess
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from opd_pilot.context_audit import require_valid_response, validate_config
from opd_pilot.environment import Episode
from tests.orion.test_integration import Array

ROOT = Path(__file__).resolve().parents[2]
REL = "trinity/common/workflows/envs/TCOD/alfworld/"
UTILS = runpy.run_path(str(ROOT / REL / "utils.py"))
SCOPE = dict(UTILS)
tree = ast.parse((ROOT / REL / "context.py").read_text())
exec(
    compile(
        ast.Module(body=[n for n in tree.body if isinstance(n, ast.FunctionDef)], type_ignores=[]),
        "context",
        "exec",
    ),
    SCOPE,
)
BUILDER = SCOPE["build_alfworld_step_messages"]


def load_workflow(source):
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    cls.bases, cls.decorator_list = [], []
    cls.body = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
        and n.name in ("reset", "_run_episode", "format_messages", "compute_reward")
    ]
    scope = dict(SCOPE, List=list, Experience=object, Task=object)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), "workflow", "exec"), scope)
    return scope[cls.name]


CURRENT = load_workflow((ROOT / REL / "OPD_workflow.py").read_text())
BASE = load_workflow(
    subprocess.check_output(
        ["git", "show", "2bf9756:" + REL + "OPD_workflow.py"], cwd=ROOT, text=True, encoding="utf-8"
    )
)


def task(mode=None):
    return NS(
        workflow_args={} if mode is None else {"context_mode": mode},
        format_args={},
        raw_task={},
        task_desc="task",
        is_eval=False,
    )


def episode(cls, mode, turns=4, instance=None, goal="TASK_SENTINEL"):
    obj = instance or cls()
    obj.reset(task(mode))
    obj.max_env_steps, obj.rollout_args, obj.temperature, obj.run_id_base = turns, {}, 1.0, 7
    messages, teacher_ids, originals = [], [], []
    observations = [f"OBS_0_SENTINEL Your task is to: {goal}"] + [
        f"OBS_{i}_SENTINEL" for i in range(1, turns + 1)
    ]
    env = NS(index=0)

    def info():
        return {"admissible_commands": [[f"cmd_{env.index}", "help"]]}

    env.reset = lambda: (observations[0], info())

    def step(action):
        assert action == f"ACTION_{env.index}_SENTINEL"
        env.index += 1
        return observations[env.index], 0.0, env.index == turns, info()

    env.step = step

    async def chat(msg, **kwargs):
        i = len(messages)
        messages.append(copy.deepcopy(msg))
        p, r = 5 + i, 3 + i
        exp = NS(
            tokens=Array(range(p + r)),
            prompt_length=p,
            logprobs=Array([-0.5] * r),
            action_mask=Array([1] * r),
            truncate_status=None,
            response_text=f"<think>THINK_{i}_SENTINEL</think><action>ACTION_{i}_SENTINEL</action>",
            metrics=None,
            eid=NS(run=0, step=0),
            reward=None,
        )
        originals.append(exp)
        return [exp]

    async def score(tokens, **kwargs):
        teacher_ids.append(tokens)
        return Array(range(len(tokens) - 1))

    obj.model, obj.teacher_model = NS(chat_async=chat), NS(logprobs_async=score)
    out = asyncio.run(obj._run_episode(env))
    return messages, out, teacher_ids, originals


class ContextTests(unittest.TestCase):
    def test_four_turn_contract_and_scoring(self):
        messages, out, ids, originals = episode(CURRENT, "per_step_h2")
        for i, (msg, exp) in enumerate(zip(messages, out)):
            self.assertEqual([m["role"] for m in msg], ["user"])
            text = msg[0]["content"]
            self.assertIn("TASK_SENTINEL", text)
            self.assertIn(f"most recent {min(i,2)} observations", text)
            for j in range(5):
                self.assertEqual(f"OBS_{j}_SENTINEL" in text, max(0, i - 2) <= j <= i)
            for j in range(4):
                self.assertNotIn(f"THINK_{j}_SENTINEL", text)
                self.assertEqual(f"ACTION_{j}_SENTINEL" in text, max(0, i - 2) <= j < i)
            self.assertIs(exp, originals[i])
            self.assertEqual(ids[i], exp.tokens.tolist())
            self.assertEqual(
                exp.teacher_logprobs, list(range(exp.prompt_length - 1, len(exp.tokens) - 1))
            )
            self.assertEqual(exp.logprobs, [-0.5] * (len(exp.tokens) - exp.prompt_length))
            self.assertEqual((exp.eid.run, exp.eid.step, exp.reward), (7, i, 1.0))
            require_valid_response(exp)

    def test_legacy_exactly_matches_prepatch(self):
        self.assertEqual(episode(BASE, None)[0], episode(CURRENT, "legacy_accumulated")[0])

    def test_reset_and_invalid_mode(self):
        obj = CURRENT()
        for mode, expected in (
            ("per_step_h2", "per_step_h2"),
            (None, "legacy_accumulated"),
            ("per_step_h2", "per_step_h2"),
        ):
            obj.reset(task(mode))
            self.assertEqual(obj.context_mode, expected)
        old = obj.task
        with self.assertRaises(ValueError):
            obj.reset(task("bad"))
        self.assertIs(obj.task, old)
        self.assertEqual(
            episode(CURRENT, "per_step_h2", instance=obj)[0],
            episode(CURRENT, "per_step_h2", instance=obj)[0],
        )

    def test_new_episode_has_new_goal(self):
        obj = CURRENT()
        episode(CURRENT, "per_step_h2", instance=obj)
        messages = episode(CURRENT, "per_step_h2", instance=obj, goal="SECOND_GOAL")[0]
        for msg in messages:
            self.assertIn("SECOND_GOAL", msg[0]["content"])
            self.assertNotIn("TASK_SENTINEL", msg[0]["content"])

    def test_long_trajectory(self):
        msgs = episode(CURRENT, "per_step_h2", 30)[0]
        self.assertTrue(all(len(msg) == 1 for msg in msgs))
        self.assertNotIn("OBS_0_SENTINEL", msgs[-1][0]["content"])
        self.assertIn("OBS_27_SENTINEL", msgs[-1][0]["content"])

    def test_builder_purity_and_evaluator_parity(self):
        history, commands = ["old", "recent1", "recent2"], [["look", "help", "open"]]
        before = copy.deepcopy((history, commands))
        expected = BUILDER(
            task_description="goal",
            observation=" current ",
            admissible_commands=commands,
            history=history,
            step_index=3,
        )
        self.assertEqual((history, commands), before)
        env = Episode.__new__(Episode)
        env.benchmark, env.context_mode, env.max_steps, env.done = (
            "alfworld",
            "per_step_h2",
            30,
            False,
        )
        env.u, env.history, env.observation, env.task_description = (
            NS(**UTILS),
            history,
            " current ",
            "goal",
        )
        env.info = {"admissible_commands": commands}
        module = NS(build_alfworld_step_messages=BUILDER)
        with patch.dict(
            sys.modules, {"trinity.common.workflows.envs.TCOD.alfworld.context": module}
        ):
            self.assertEqual(env.prompt(), expected)

    def test_eval_mode_must_be_explicit(self):
        import yaml

        config = yaml.safe_load(
            (ROOT / "orion/generated/alfworld-s17-context-h2-smoke-first/config.yaml").read_text()
        )
        self.assertTrue(validate_config(config))
        config["buffer"]["explorer_input"]["eval_tasksets"] = [{"workflow_args": {}}]
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_real_experience_masks_and_padding(self):
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            self.skipTest("CPU PyTorch is required for the real mask check")
        from trinity.common.experience import Experience, gather_action_masks

        normal = Experience(tokens=[10, 11, 12, 13, 14], prompt_length=2, logprobs=[-1, -2, -3])
        short = Experience(tokens=[20, 21], prompt_length=1, logprobs=[-1])
        dummy = Experience(
            tokens=[30, 31], prompt_length=1, logprobs=[0], truncate_status="prompt_truncated"
        )
        masks = gather_action_masks([normal, short, dummy], 3).tolist()
        self.assertEqual(masks, [[True, True, True], [True, False, False], [False, False, False]])
        require_valid_response(normal)
        with self.assertRaises(ValueError):
            require_valid_response(dummy)

    def test_dummy_guard_prevents_action(self):
        from orion.scripts.alfworld_context_smoke import GuardedStudent

        dummy = NS(
            tokens=Array([1, 2]),
            prompt_length=1,
            logprobs=Array([0]),
            action_mask=Array([0]),
            truncate_status="prompt_truncated",
            response_text="dummy",
        )

        async def chat(*args, **kwargs):
            return [dummy]

        rows = []

        async def call():
            await GuardedStudent(NS(chat_async=chat), rows, 10240, 20480).chat_async(
                [{"role": "user", "content": "x"}]
            )
            self.fail("dummy was allowed to reach action execution")

        with self.assertRaises(ValueError):
            asyncio.run(call())
        self.assertEqual(rows[0]["truncate_status"], "prompt_truncated")


if __name__ == "__main__":
    unittest.main()
