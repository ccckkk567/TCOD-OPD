"""Check lossless cross-worker merge and rejection of invalid evidence."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from opd_pilot.merge_records import merge
from opd_pilot.records import Recorder, audit


class MergeRecords(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.source = root / "source"
        self.destination = root / "combined"
        for index in range(2):
            with Recorder(self.source, f"worker-{index}") as writer:
                writer.record("episode", episode_uid=f"episode-{index}", task_id="task", outcome={})

    def test_merge_preserves_original_lines_and_sources(self):
        originals = [path.read_bytes() for path in sorted(self.source.glob("*/events.jsonl"))]
        result = merge(self.source, self.destination)
        self.assertEqual(result["records"]["episode"], 2)
        self.assertEqual(
            (self.destination / "merged/events.jsonl").read_bytes(), b"".join(originals)
        )
        self.assertEqual(audit(self.destination)["records"], audit(self.source)["records"])
        self.assertEqual(
            originals, [p.read_bytes() for p in sorted(self.source.glob("*/events.jsonl"))]
        )
        with self.assertRaises(FileExistsError):
            merge(self.source, self.destination)

    def test_invalid_source_and_nested_destination_rejected(self):
        with self.assertRaises(ValueError):
            merge(self.source, self.source / "combined")
        (self.source / "worker-0/COMPLETE.json").unlink()
        with self.assertRaises(FileNotFoundError):
            merge(self.source, self.destination)
        self.assertFalse(self.destination.exists())

    def test_write_failure_does_not_publish_destination(self):
        with patch("opd_pilot.merge_records.shutil.copyfileobj", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                merge(self.source, self.destination)
        self.assertFalse(self.destination.exists())
        self.assertTrue(self.destination.with_name("combined.in-progress").is_dir())
        self.assertEqual(audit(self.source)["records"]["episode"], 2)
