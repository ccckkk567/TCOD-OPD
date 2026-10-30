"""Do not manufacture complete paired comparisons from missing branches."""

import unittest

from opd_pilot.branch_analysis import summarize


class BranchAnalysis(unittest.TestCase):
    def rows(self):
        return [
            dict(
                prefix_uid="prefix",
                task_id="task",
                candidate_owner=owner,
                executor=executor,
                candidate_uid=owner + "-candidate",
                repeat=0,
                replay_validated=True,
                outcome=dict(technical_failure=False, raw_score=value),
            )
            for owner, executor, value in (
                ("student", "student", 0.8),
                ("student", "teacher", 0.2),
                ("teacher", "student", 0.3),
                ("teacher", "teacher", 0.9),
            )
        ]

    def test_paired_reversal_and_missing_never_zero(self):
        rows = self.rows()
        result = summarize(rows, "raw_score", True)
        self.assertEqual(result["complete_prefixes"], 1)
        self.assertEqual(result["exploratory_reversal_fraction"], 1)
        result = summarize(rows[:-1], "raw_score", True)
        self.assertEqual(result["complete_prefixes"], 0)
        self.assertIsNone(result["exploratory_reversal_fraction"])

    def test_duplicates_and_failed_replays_are_not_used(self):
        rows = self.rows()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(rows + rows[:1], "raw_score", True)
        rows[0]["replay_validated"] = False
        result = summarize(rows, "raw_score", False)
        self.assertEqual(result["counts"]["replay_unverified"], 1)
        self.assertEqual(result["complete_prefixes"], 0)
