"""Verify restricted mirror provenance and simulator-smoke cleanup without downloads."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from orion.scripts.benchmark_runtime_smoke import scienceworld
from orion.scripts.download_webshop import FILES, MIRROR_REVISION, download


class BenchmarkRecovery(unittest.TestCase):
    def test_mirror_fetches_only_three_fixed_files_and_cannot_be_relabeled_official(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calls = []

            def fetch(**kwargs):
                calls.append(kwargs)
                (root / kwargs["filename"]).write_text(json.dumps([{"fixture": True}]))

            with patch.dict(
                "sys.modules", {"huggingface_hub": SimpleNamespace(hf_hub_download=fetch)}
            ):
                manifest = download(root, "hf-smoke")
                download(root, "hf-smoke")
            self.assertEqual({call["filename"] for call in calls}, set(FILES))
            self.assertEqual(len(calls), 3)
            self.assertTrue(all(call["revision"] == MIRROR_REVISION for call in calls))
            self.assertEqual(manifest["usage_scope"], "small_environment_smoke_only")
            with self.assertRaisesRegex(ValueError, "relabel"):
                download(root, "official")

    def test_scienceworld_smoke_uses_original_factory_and_closes_on_error(self):
        env = Mock()
        env.step.return_value = ("next", 0, False, {"score": 0})
        helpers = {
            "_create_scienceworld_env": Mock(return_value=env),
            "_reset_scienceworld_env": Mock(return_value=("room", {"score": 0}, "task")),
            "_get_compact_action_info": Mock(return_value=(["look around"], ["room"])),
        }
        rows = scienceworld(Path("scienceworld.jar"), helpers)
        self.assertEqual([row["task"]["task_name"] for row in rows], ["boil", "melt"])
        self.assertEqual(env.close.call_count, 2)
        env.step.side_effect = RuntimeError("Java failed")
        with self.assertRaisesRegex(RuntimeError, "Java failed"):
            scienceworld(Path("scienceworld.jar"), helpers)
        self.assertEqual(env.close.call_count, 3)
