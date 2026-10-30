"""Check reuse and interrupted downloads without downloading model weights."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from opd_pilot.assets import download_model


class ModelAssets(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.contents = {
            "config.json": b"{}",
            "tokenizer.json": b"{}",
            "tokenizer_config.json": b"{}",
            "model.safetensors.index.json": json.dumps(
                {"weight_map": {"weight": "model-00001-of-00001.safetensors"}}
            ).encode(),
            "model-00001-of-00001.safetensors": b"test-weight",
        }
        self.api = Mock()
        self.api.model_info.return_value = SimpleNamespace(
            sha="fixed-revision",
            siblings=[SimpleNamespace(rfilename=k, size=len(v)) for k, v in self.contents.items()],
        )

    def fetch(self, **kwargs):
        for name, data in self.contents.items():
            (Path(kwargs["local_dir"]) / name).write_bytes(data)

    def test_ready_model_is_reused_without_network(self):
        download = Mock(side_effect=self.fetch)
        path = download_model("s17", self.root, self.api, download)
        download_model("s17", self.root, self.api, download)
        self.assertEqual(path.name, "Qwen3-1.7B")
        self.api.model_info.assert_called_once()
        download.assert_called_once()

    def test_interrupted_download_reuses_pinned_revision(self):
        with self.assertRaises(ConnectionError):
            download_model("s17", self.root, self.api, Mock(side_effect=ConnectionError))
        self.assertFalse((self.root / "Qwen3-1.7B/READY.json").exists())
        download = Mock(side_effect=self.fetch)
        download_model("s17", self.root, self.api, download)
        self.assertEqual(download.call_args.kwargs["revision"], "fixed-revision")
        self.api.model_info.assert_called_once()

    def test_missing_shard_does_not_publish_ready(self):
        with self.assertRaisesRegex(ValueError, "Missing or incomplete"):
            download_model("s17", self.root, self.api, Mock())
        self.assertFalse((self.root / "Qwen3-1.7B/READY.json").exists())

    def test_existing_unmanaged_files_are_not_overwritten(self):
        folder = self.root / "Qwen3-1.7B"
        folder.mkdir()
        (folder / "config.json").write_text("existing")
        with self.assertRaisesRegex(ValueError, "unmanaged"):
            download_model("s17", self.root, self.api, Mock())
        self.api.model_info.assert_not_called()
        self.assertEqual((folder / "config.json").read_text(), "existing")
