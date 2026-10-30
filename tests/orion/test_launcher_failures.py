"""Exercise production launcher bodies with fake Ray, without a training installation."""

import ast
import subprocess
import sys
import traceback
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

SOURCE = Path(__file__).resolve().parents[2] / "trinity/cli/launcher.py"
FUNCTIONS = {"bench", "explore", "train", "serve", "both", "_wait_for_completion"}


def load_launcher():
    """Load actual function bodies; omit heavyweight framework imports only."""
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in FUNCTIONS]
    namespace = {
        "Config": object,
        "ray": Mock(),
        "Explorer": Mock(),
        "Trainer": Mock(),
        "logger": Mock(),
        "traceback": traceback,
    }
    for role in ("Explorer", "Trainer"):
        namespace[role].get_actor.return_value.__ray_ready__ = Mock()
    exec(compile(tree, str(SOURCE), "exec"), namespace)
    return namespace


def config():
    return SimpleNamespace(
        explorer=SimpleNamespace(name="explorer"),
        trainer=SimpleNamespace(name="trainer"),
        synchronizer=SimpleNamespace(sync_timeout=30),
    )


class LauncherFailures(unittest.TestCase):
    def test_single_role_errors_escape(self):
        for mode in ("bench", "train", "explore", "serve"):
            with self.subTest(mode=mode):
                ns = load_launcher()
                ns["ray"].get.side_effect = RuntimeError("remote failure")
                with self.assertRaisesRegex(RuntimeError, "remote failure"):
                    ns[mode](config())

    def test_single_role_success(self):
        for mode in ("bench", "train", "explore", "serve"):
            ns = load_launcher()
            ns[mode](config())

    def test_late_remote_failure_is_retrieved(self):
        ns = load_launcher()
        ns["ray"].wait.return_value = (["failed-ref"], [])
        ns["ray"].get.side_effect = RuntimeError("late failure")
        with self.assertRaisesRegex(RuntimeError, "late failure"):
            ns["_wait_for_completion"](["failed-ref"], 30)

    def test_required_timeout(self):
        ns = load_launcher()
        ns["ray"].wait.return_value = ([], ["trainer-ref"])
        with self.assertRaises(TimeoutError):
            ns["_wait_for_completion"](["trainer-ref"], 30)

    def test_explorer_grace_period_can_expire(self):
        ns = load_launcher()
        ns["ray"].wait.return_value = ([], ["explorer-ref"])
        ns["_wait_for_completion"](["explorer-ref"], 5, required=False)

    def test_explorer_finish_does_not_hide_trainer_timeout(self):
        ns = load_launcher()
        ns["ray"].get.side_effect = [None, None, None, "explorer", None]
        ns["ray"].wait.side_effect = [(["explorer-ref"], ["trainer-ref"]), ([], ["trainer-ref"])]
        with self.assertRaises(TimeoutError):
            ns["both"](config())

    def test_trainer_finish_surfaces_shutdown_failure(self):
        ns = load_launcher()
        ns["ray"].get.side_effect = [
            None,
            None,
            None,
            "trainer",
            None,
            RuntimeError("shutdown failed"),
        ]
        ns["ray"].wait.side_effect = [
            (["trainer-ref"], ["explorer-ref"]),
            ([], ["explorer-ref"]),
            (["shutdown-explorer", "shutdown-trainer"], []),
        ]
        with self.assertRaisesRegex(RuntimeError, "shutdown failed"):
            ns["both"](config())

    def test_unhandled_failure_returns_nonzero(self):
        code = (
            "from test_launcher_failures import load_launcher, config; "
            "ns=load_launcher(); "
            "ns['ray'].get.side_effect=RuntimeError('remote failure'); "
            "ns['train'](config())"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=Path(__file__).parent, capture_output=True, text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("remote failure", result.stderr)


if __name__ == "__main__":
    unittest.main()
