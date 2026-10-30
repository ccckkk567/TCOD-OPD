"""Monitor failure checks using the production class with a fake SwanLab SDK."""

import ast
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import Mock, patch

from trinity.utils.monitor_helpers import redact_secrets


def load_monitor(sdk):
    source = Path(__file__).resolve().parents[2] / "trinity/utils/monitor.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SwanlabMonitor"]
    namespace = {
        "Monitor": object,
        "Config": object,
        "Dict": Dict,
        "List": List,
        "pd": SimpleNamespace(DataFrame=object),
        "swanlab": sdk,
        "Path": Path,
        "os": os,
        "json": json,
        "redact_secrets": redact_secrets,
        "get_logger": Mock(return_value=Mock()),
    }
    exec(compile(tree, str(source), "exec"), namespace)
    return namespace["SwanlabMonitor"]


class SwanlabModes(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.sdk = Mock()
        self.cls = load_monitor(self.sdk)

    def monitor(self, mode="offline", role="trainer"):
        cfg = SimpleNamespace(
            monitor=SimpleNamespace(monitor_args={"mode": mode, "logdir": self.temp.name}),
            flatten=lambda: {"monitor.api_key": "private-key", "max_tokens": 512},
        )
        instance = self.cls("pilot", "group", "run", role, cfg)
        self.addCleanup(instance.close)
        return instance

    def test_offline_without_key_and_instance_logging(self):
        with patch.dict(os.environ, {}, clear=True):
            monitor = self.monitor()
        self.sdk.login.assert_not_called()
        self.assertEqual(self.sdk.init.call_args.kwargs["mode"], "offline")
        self.assertEqual(self.sdk.init.call_args.kwargs["config"]["monitor.api_key"], "[REDACTED]")
        monitor.log({"loss": 0.5}, 1)
        self.sdk.init.return_value.log.assert_called_once_with({"loss": 0.5}, step=1)
        self.sdk.log.assert_not_called()
        monitor.close()
        monitor.close()
        self.sdk.init.return_value.finish.assert_called_once()

    def test_login_failure_falls_back_without_saving_key(self):
        self.sdk.login.side_effect = ConnectionError("network failure")
        with patch.dict(os.environ, {"SWANLAB_API_KEY": "private-key"}, clear=True):
            self.monitor("cloud")
        self.sdk.login.assert_called_once_with(api_key="private-key", save=False)
        self.assertEqual(self.sdk.init.call_args.kwargs["mode"], "offline")

    def test_sdk_failure_keeps_local_evidence(self):
        self.sdk.init.side_effect = RuntimeError("SDK unavailable")
        monitor = self.monitor()
        monitor.log({"loss": 0.25}, 3)
        monitor.close()
        path = next(Path(self.temp.name).rglob("metrics-*.jsonl"))
        records = [json.loads(line) for line in path.read_text().splitlines()]
        self.assertEqual(records[-1]["data"], {"loss": 0.25})

    def test_roles_have_separate_logs(self):
        a, b = self.monitor(role="trainer"), self.monitor(role="explorer")
        self.assertNotEqual(a._metrics.name, b._metrics.name)
        c = self.monitor(role="trainer")
        self.assertNotEqual(a._metrics.name, c._metrics.name)

    def test_runtime_keys_are_redacted_but_token_counts_remain(self):
        with patch.dict(os.environ, {"HF_TOKEN": "test-secret"}):
            result = redact_secrets({"message": "test-secret", "max_tokens": 10})
        self.assertEqual(result, {"message": "[REDACTED]", "max_tokens": 10})
