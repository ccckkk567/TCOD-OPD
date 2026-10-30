"""Check dependency validation before submission and preserve scheduler failures."""

import tempfile
import unittest
from pathlib import Path

from opd_pilot.jobs import order


class Jobs(unittest.TestCase):
    def test_order_and_cycle(self):
        with tempfile.TemporaryDirectory() as directory:
            spec = Path(directory).resolve() / "input.json"
            spec.write_text("{}")
            tasks = [
                dict(
                    name="branches",
                    stage="continuation",
                    spec=str(spec),
                    time="01:00:00",
                    after=["candidates"],
                ),
                dict(name="candidates", stage="candidates", spec=str(spec), time="00:30:00"),
            ]
            self.assertEqual([task["name"] for task in order(tasks)], ["candidates", "branches"])
            tasks[1]["after"] = ["branches"]
            with self.assertRaisesRegex(ValueError, "cycle"):
                order(tasks)

    def test_duration_limit_before_submission(self):
        with self.assertRaisesRegex(ValueError, "18 hours"):
            order([dict(name="long", stage="evaluate", spec="unused", time="18:00:01")])
