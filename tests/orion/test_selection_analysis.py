"""Keep measured adoption separate from likelihood and fixed-candidate execution."""

import json
import tempfile
import unittest
from pathlib import Path

from opd_pilot.selection_analysis import adoption, likelihoods


class SelectionAnalysis(unittest.TestCase):
    def test_adoption_uses_actual_action_samples_and_likelihood_checks_same_response(self):
        candidates = {
            owner: dict(
                candidate_uid=owner,
                prefix_uid="prefix",
                task_id="task",
                owner=owner,
                prefix_tokens=[1],
                response_tokens=[2],
                parsed_action="shared-action",
            )
            for owner in ("student", "teacher")
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generated = root / "generated.json"
            generated.write_text(
                json.dumps(
                    dict(
                        candidates=[
                            dict(
                                prefix_uid="prefix",
                                prefix_tokens=[1],
                                owner="student",
                                generation_revision="step25",
                                parsed_action="shared-action",
                            )
                            for _ in range(2)
                        ]
                    )
                )
            )
            entry = dict(revision="step25", generated_banks=[str(generated)])
            frequency = adoption(entry, candidates)["prefix"]
            self.assertEqual(frequency["adoption_samples"], 2)
            self.assertEqual(frequency["teacher_action_frequency"], 1)
            self.assertEqual(frequency["overlapping_action_frequency"], 1)
            path = root / "scores.json"
            bank = dict(
                scoring_revision="step25",
                temperature=1,
                scores=[
                    dict(
                        candidate_uid=owner,
                        prefix_uid="prefix",
                        response_tokens=[2],
                        mean_token_nll=3,
                    )
                    for owner in candidates
                ],
            )
            path.write_text(json.dumps(bank))
            entry["scores"] = [str(path)]
            self.assertEqual(
                likelihoods(entry, {"temperature": 1}, candidates)[("prefix", "teacher")], [3]
            )
            bank["scores"][0]["response_tokens"] = [99]
            path.write_text(json.dumps(bank))
            with self.assertRaisesRegex(ValueError, "fixed candidate"):
                likelihoods(entry, {"temperature": 1}, candidates)
