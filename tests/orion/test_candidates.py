"""Check exact diagnostic prefixes, seed independence and conditional score contracts."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from opd_pilot.candidates import generate, score, validate
from opd_pilot.diagnostic_backend import (
    prediction_positions,
    sample_seed,
    token_identity,
)


class Candidates(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tokenizer.json").write_text(
            json.dumps(
                {
                    "model": {"type": "BPE", "vocab": {"a": 0, "b": 1}},
                    "added_tokens": [{"id": 2, "content": "<eos>"}],
                }
            )
        )
        self.spec = dict(
            model_path=str(self.root),
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=10,
            owner="student",
            samples_per_prefix=2,
            seed=42,
        )
        self.prefix = dict(
            prefix_uid="prefix-0001",
            task_id="task-1",
            source_turn_uid="turn-1",
            messages=[{"role": "user", "content": "actual prompt"}],
            prefix_tokens=[0, 1],
            sampling_student_revision="student-v1",
        )

    def test_generation_preserves_actual_prefix_and_duplicate_samples(self):
        backend = SimpleNamespace(
            generate=Mock(
                return_value={
                    "response_tokens": [1, 2],
                    "response_text": "b",
                    "termination": "eos",
                }
            )
        )
        result = generate(
            self.spec, {"revision": "student-v1"}, {"prefixes": [self.prefix]}, backend
        )
        self.assertEqual(len(result["candidates"]), 2)
        self.assertEqual(backend.generate.call_args_list[0].args[0], [0, 1])
        self.assertEqual(result["candidates"][0]["messages"], self.prefix["messages"])
        self.assertNotEqual(result["candidates"][0]["seed"], result["candidates"][1]["seed"])
        self.assertEqual(
            token_identity(self.root), json.loads(json.dumps(result["token_identity"]))
        )
        self.assertFalse(result["environment_execution_complete"])

    def test_scoring_retains_exact_ids_and_reports_both_nlls(self):
        backend = SimpleNamespace(score=Mock(return_value=[-0.2, -0.8]))
        bank = {
            "candidates": [
                {
                    "candidate_uid": "student-prefix-1",
                    "prefix_uid": "prefix-1",
                    "prefix_tokens": [0, 1],
                    "response_tokens": [1, 2],
                }
            ]
        }
        result = score(self.spec, {"revision": "teacher-v1"}, bank, backend)
        backend.score.assert_called_once_with([0, 1], [1, 2], 1.0)
        self.assertEqual(result["scores"][0]["sum_nll"], 1)
        self.assertEqual(result["scores"][0]["mean_token_nll"], 0.5)
        backend.score.return_value = [-0.2]
        with self.assertRaises(ValueError):
            score(self.spec, {"revision": "teacher-v1"}, bank, backend)

    def test_response_positions_and_identity_seeds(self):
        self.assertEqual(prediction_positions([10, 20, 30], [40, 50]), [2, 3])
        self.assertEqual(
            sample_seed(42, "prefix-1", "teacher", 0), sample_seed(42, "prefix-1", "teacher", 0)
        )
        self.assertNotEqual(
            sample_seed(42, "prefix-1", "teacher", 0), sample_seed(42, "prefix-2", "teacher", 0)
        )

    def test_cross_tokenizer_bank_is_rejected_before_gpu_loading(self):
        bank = self.root / "input.json"
        bank.write_text(
            json.dumps(
                {
                    "token_identity": {"wrong": "mapping"},
                    "candidates": [{"candidate_uid": "candidate-1"}],
                }
            )
        )
        spec = dict(
            self.spec, operation="score", prefix_tokenizer_path=str(self.root), input_bank=str(bank)
        )
        with patch("opd_pilot.candidates.model_manifest", return_value={"revision": "student-v1"}):
            with self.assertRaisesRegex(ValueError, "token identity"):
                validate(spec)
