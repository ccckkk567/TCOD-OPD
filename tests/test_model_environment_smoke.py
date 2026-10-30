"""Failure and alignment checks for the independent one-GPU diagnostic."""

import unittest
from unittest.mock import Mock, patch

from orion.scripts.model_environment_smoke import interact, score_turns


class ModelEnvironmentSmokeTests(unittest.TestCase):
    def test_generation_failure_closes_simulator(self):
        episode = Mock()
        backend = Mock()
        backend.generate.side_effect = RuntimeError("generation failed")
        with patch("orion.scripts.model_environment_smoke.Episode", return_value=episode):
            with self.assertRaisesRegex(RuntimeError, "generation failed"):
                interact("alfworld", ["task"], Mock(), backend)
        episode.close.assert_called_once()
        episode.apply.assert_not_called()

    def test_teacher_scores_exact_student_prefix_and_response(self):
        turn = dict(prefix_tokens=[1, 2], response_tokens=[3, 4])
        backend = Mock()
        backend.score.return_value = [-0.2, -0.3]
        self.assertEqual(score_turns([dict(turns=[turn])], backend), 1)
        backend.score.assert_called_once_with([1, 2], [3, 4], 1.0)
        self.assertEqual(turn["teacher_logprobs"], [-0.2, -0.3])

    def test_incomplete_teacher_scores_fail(self):
        backend = Mock()
        backend.score.return_value = [-0.2]
        with self.assertRaisesRegex(RuntimeError, "align"):
            score_turns([dict(turns=[dict(prefix_tokens=[1], response_tokens=[2, 3])])], backend)


if __name__ == "__main__":
    unittest.main()
