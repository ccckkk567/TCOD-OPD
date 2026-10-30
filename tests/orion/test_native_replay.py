"""Native tensor envelopes and real optimizer counting; no distributed GPU claims."""

import ast
import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

import torch

from opd_pilot.native_codec import capture, decode, encode, restore
from opd_pilot.native_replay import verify_workers
from opd_pilot.replay_observer import ActorTrace, single_step_errors
from opd_pilot.update_probe import digest
from trinity.common.experience import Experience


class NativeReplay(unittest.TestCase):
    def test_native_actor_loop_has_same_cpu_update_with_observer_on_and_off(self):
        source = Path(__file__).resolve().parents[2] / "trinity/trainer/verl/dp_actor.py"
        tree = ast.parse(source.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
        method = next(
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef) and node.name == "update_policy"
        )
        method.decorator_list = []

        class ZeroLoss:
            def __call__(self, **kwargs):
                return 0.0, {}

            def calculate_kl_loss(self, **kwargs):
                return 0.0, {}

        class Data:
            def __init__(self, tensors):
                self.batch, self.non_tensor_batch = tensors, {}
                self.meta_info = {"temperature": 1.0}

            def __len__(self):
                return len(self.batch["responses"])

            def select(self, batch_keys, non_tensor_batch_keys):
                return Data({k: self.batch[k] for k in batch_keys})

            def split(self, size):
                return [
                    Data({k: v[i : i + size] for k, v in self.batch.items()})
                    for i in range(0, len(self), size)
                ]

            def to(self, device):
                return self

        scope = dict(
            DataProto=Data,
            torch=torch,
            DummyKLFn=ZeroLoss,
            DummyEntropyLossFn=ZeroLoss,
            get_device_id=lambda: "cpu",
            prefix_metrics=lambda **kwargs: None,
            append_to_dict=lambda destination, values: destination.update(values),
        )
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), scope)
        outcomes = []
        for recording in (False, True):
            parameter = torch.nn.Parameter(torch.tensor(1.0))
            optimizer = torch.optim.AdamW([parameter], lr=0.01)

            class PolicyLoss:
                select_keys = []

                def __call__(self, logprob, **kwargs):
                    return logprob.mean(), {}

            actor = NS(
                actor_module=NS(train=lambda: None),
                actor_optimizer=optimizer,
                config=NS(
                    ppo_mini_batch_size=2,
                    ppo_epochs=1,
                    use_dynamic_bsz=False,
                    ppo_micro_batch_size_per_gpu=1,
                    fix_actor_microbatch_loss_scale=False,
                    policy_loss={},
                ),
                loss_agg_mode="token-mean",
                policy_loss_fn=PolicyLoss(),
                kl_loss_fn=ZeroLoss(),
                entropy_loss_fn=ZeroLoss(),
                scaler=None,
            )
            actor._forward_micro_batch = lambda **kwargs: (None, parameter.expand(1, 1))
            calls = []

            def step():
                calls.append("optimizer")
                optimizer.step()
                return torch.tensor(1.0)

            actor._optimizer_step = step
            trace = (
                NS(attempts=0, micro_order=lambda order: calls.append(order)) if recording else None
            )
            actor._e5_replay_trace = trace
            data = Data(
                {
                    k: torch.ones(2, 1)
                    for k in (
                        "input_ids",
                        "position_ids",
                        "attention_mask",
                        "responses",
                        "response_mask",
                    )
                }
            )
            scope["update_policy"](actor, data)
            self.assertEqual(calls.count("optimizer"), 1)
            if trace:
                self.assertEqual(trace.attempts, 1)
                self.assertIn([[0], [1]], calls)
            outcomes.append(parameter.detach().clone())
        self.assertTrue(torch.equal(*outcomes))

    def test_native_roundtrip_keeps_dtype_dynamic_fields_and_mask(self):
        exp = Experience(
            tokens=torch.tensor([1, 2, 3], dtype=torch.int64),
            prompt_length=1,
            logprobs=torch.tensor([-1, -2], dtype=torch.float64),
            reward=0,
            teacher_logprobs=torch.tensor([-0.5, -0.7]),
            info={"model_version": 25},
        )
        exp.teacher_logprobs_valid_mask = torch.tensor([True, True])
        exp.extra_metadata = ("tuple", torch.tensor([1], dtype=torch.int16))
        batch = capture(
            [exp],
            dict(
                purpose="e5_fresh_student_train",
                official_split="train",
                student_revision="step25",
                checkpoint_revision="step25",
                sampling={"temperature": 1},
            ),
        )
        scores = dict(
            batch_digest=digest(batch),
            teacher_revision="teacher",
            tokenizer_revision="tok",
            backend="native",
            temperature=1,
            rows=copy.deepcopy(batch["rows"]),
        )
        scores["rows"][0].update(
            teacher_log_probs=exp.teacher_logprobs.tolist(), valid_mask=[True, True]
        )
        restored = restore(batch, scores)[0]
        self.assertEqual(encode(vars(exp)), encode(vars(restored)))
        self.assertEqual(restored.logprobs.dtype, torch.float64)
        self.assertIsInstance(restored.extra_metadata, tuple)
        self.assertEqual(decode(encode(torch.bfloat16)), torch.bfloat16)
        batch["rows"][0]["native_fields"]["items"].append(["prompt_length", 2])
        scores["batch_digest"] = digest(batch)
        with self.assertRaisesRegex(ValueError, "roundtrip|disagrees"):
            restore(batch, scores)

    def test_single_step_preflight_does_not_change_batch_or_epochs(self):
        self.assertEqual(single_step_errors(8, 8, 1, 1), [])
        self.assertTrue(single_step_errors(16, 8, 1, 1))
        self.assertTrue(single_step_errors(8, 8, 2, 1))
        self.assertTrue(single_step_errors(8, 8, 1, 2))

    def test_actual_optimizer_hook_counts_step_and_detects_skipped_attempt(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        actor = NS(
            actor_optimizer=optimizer,
            config=NS(ppo_mini_batch_size=1, ppo_epochs=1),
            ulysses_sequence_parallel_size=1,
        )

        class Data:
            non_tensor_batch = {"unique_ids": NS(tolist=lambda: ["sample"])}

            def __len__(self):
                return 1

        with patch("torch.distributed.get_rank", return_value=0), patch(
            "opd_pilot.replay_observer.collective_require"
        ):
            trace = ActorTrace(actor, Data())
            trace.attempts += 1
            self.assertEqual(trace.executed, 0)
            model(torch.ones(1, 1)).sum().backward()
            optimizer.step()
            self.assertEqual(trace.executed, 1)
            trace.micro_order([[0]])
            with patch("opd_pilot.replay_observer.write") as writer:
                trace.finish(25, 26)
                row = writer.call_args.args[1]
            self.assertEqual(row["micro_batches"], [["sample"]])
            self.assertEqual(row["scheduler_steps"], 1)

    def test_rank_counts_are_one_logical_step_not_sum(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = dict(
                status="returned",
                optimizer_step_attempts=1,
                optimizer_steps_executed=1,
                scheduler_steps=1,
            )
            for rank in (0, 1):
                (root / f"rank-{rank}-update.json").write_text(json.dumps(row))
            self.assertEqual(len(verify_workers(root, 2)), 2)
            row["optimizer_steps_executed"] = 0
            (root / "rank-1-update.json").write_text(json.dumps(row))
            with self.assertRaisesRegex(RuntimeError, "exactly one"):
                verify_workers(root, 2)
