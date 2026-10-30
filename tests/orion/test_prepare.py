"""Ensure pilot lists do not overlap or consume official evaluation games."""

import json
import tempfile
import unittest
from pathlib import Path

from opd_pilot.prepare import prepare_alfworld, validate_splits


class PilotSplits(unittest.TestCase):
    def test_disjoint_reproducible_and_proportional(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for category, count in (("pick_and_place_simple", 12), ("pick_clean_then_place", 6)):
                for number in range(count):
                    game = (
                        root
                        / "data/json_2.1.1/train"
                        / f"{category}-object"
                        / str(number)
                        / "game.tw-pddl"
                    )
                    game.parent.mkdir(parents=True, exist_ok=True)
                    game.write_text("fixture")
            counts = {"train_pilot": 6, "analysis_dev": 3, "confirm_holdout": 3, "smoke": 3}
            for folder in ("first", "second"):
                protocol = prepare_alfworld(root / "data", root / folder, counts=counts)
                self.assertFalse(protocol["official_test_used"])
                self.assertFalse(protocol["teacher_success_filter"])
                validate_splits(root / folder)
            for name in counts:
                self.assertEqual(
                    (root / "first" / f"{name}.jsonl").read_bytes(),
                    (root / "second" / f"{name}.jsonl").read_bytes(),
                )
            train = root / "first/train_pilot.jsonl"
            dev = root / "first/analysis_dev.jsonl"
            rows = dev.read_text().splitlines()
            rows[0] = train.read_text().splitlines()[0]
            dev.write_text("\n".join(rows) + "\n")
            with self.assertRaisesRegex(ValueError, "Duplicated"):
                validate_splits(root / "first")

    def test_small_pool_is_scaled_without_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            game = root / "data/json_2.1.1/train/pick_and_place_simple-object/0/game.tw-pddl"
            game.parent.mkdir(parents=True)
            game.write_text("fixture")
            protocol = prepare_alfworld(root / "data", root / "splits")
            self.assertEqual(sum(protocol["actual_counts"].values()), 1)
            self.assertEqual(
                json.loads((root / "splits/SPLITS_READY.json").read_text())["seed"], 42
            )
