"""Cover the observed malformed header and keep installed-file metadata consistent."""

import csv
import tempfile
import unittest
from email.parser import Parser
from pathlib import Path

from orion.scripts.repair_textworld_metadata import normalized_wheel, repair


class TextWorldMetadata(unittest.TestCase):
    def test_observed_separator_repair_preserves_tag_and_updates_record(self):
        text = "Wheel-Version: 1.0\nRoot-Is-Purelib: false\n\nTag: py3-none-manylinux2014_x86_64\n"
        self.assertIsNone(Parser().parsestr(text).get_all("Tag"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dist = root / "textworld-1.7.0.dist-info"
            dist.mkdir()
            wheel, record = dist / "WHEEL", dist / "RECORD"
            wheel.write_text(text, encoding="utf-8")
            record.write_text(dist.name + "/WHEEL,,\ntextworld/__init__.py,sha256=unchanged,12\n")
            result = repair(
                wheel, root / "backup", lambda tags: tags == ["py3-none-manylinux2014_x86_64"]
            )
            self.assertEqual(result["status"], "metadata_repaired")
            with record.open(newline="") as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(int(rows[0][2]), wheel.stat().st_size)
            self.assertEqual(rows[1], ["textworld/__init__.py", "sha256=unchanged", "12"])
            self.assertEqual(normalized_wheel(wheel.read_text()), wheel.read_text())
            self.assertEqual((root / "backup/WHEEL.before").read_text(), text)

    def test_unexpected_body_is_not_silently_removed(self):
        with self.assertRaises(ValueError):
            normalized_wheel("Wheel-Version: 1.0\n\nUnexpected metadata body\n")
