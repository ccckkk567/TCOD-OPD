"""Failure and alignment checks for the independent one-GPU diagnostic."""

import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from opd_pilot.diagnostic_backend import TransformersBackend
from opd_pilot.integration import plain

from orion.scripts.environment_interface_smoke import json_value
from orion.scripts.model_environment_smoke import action_summary, interact, score_turns


class ModelEnvironmentSmokeTests(unittest.TestCase):
    def test_nested_textworld_facts_preserved(self):
        # Match TextWorld's public serialize contract without installing a simulator locally.
        proposition = type(
            "Proposition",
            (),
            {
                "__module__": "textworld.logic",
                "serialize": lambda self: {
                    "name": "open",
                    "arguments": [{"name": "cabinet", "type": "c"}],
                },
            },
        )()
        encoded = json.loads(json.dumps({"info": {"facts": [[proposition]]}}, default=json_value))
        fact = encoded["info"]["facts"][0][0]
        self.assertEqual(fact["object_type"], "textworld.logic.Proposition")
        self.assertEqual(fact["data"]["arguments"][0]["name"], "cabinet")
        self.assertEqual(json.loads(json.dumps(plain({"facts": [proposition]})))["facts"][0], fact)
        with self.assertRaises(TypeError):
            json.dumps(object(), default=json_value)

    def test_empty_actions_are_not_a_clean_pass(self):
        summary = action_summary(
            [dict(turns=[dict(transition=dict(action=action)) for action in ["look", "", " "]])]
        )
        self.assertEqual(summary["empty_parsed_actions"], 2)
        self.assertFalse(summary["all_actions_nonempty"])

    def test_sampling_arguments_override_model_defaults(self):
        from unittest.mock import MagicMock

        backend = TransformersBackend.__new__(TransformersBackend)
        backend.torch = MagicMock()
        backend.tokenizer = Mock(pad_token_id=0)
        backend.model = MagicMock()
        backend.model.generation_config.eos_token_id = 9
        backend.model.generate.return_value.__getitem__.return_value.tolist.return_value = [3, 9]
        settings = dict(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=512)
        with patch.dict(
            "sys.modules", {"transformers": SimpleNamespace(GenerationConfig=SimpleNamespace)}
        ):
            backend.generate([1, 2], settings, 42)
        kwargs = backend.model.generate.call_args.kwargs
        for key, expected in settings.items():
            self.assertEqual(kwargs[key], expected)

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
