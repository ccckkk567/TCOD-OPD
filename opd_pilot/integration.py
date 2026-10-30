"""Opt-in observation hooks for approved TCOD benchmark and actor-update call sites."""

import os
import socket
from contextlib import nullcontext
from copy import deepcopy
from itertools import count

from opd_pilot.records import Recorder

SESSIONS = count(1)


def plain(value):
    """Copy tensor/NumPy evidence into JSON-compatible CPU values."""
    if hasattr(value, "detach"):
        value = value.detach().cpu()
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if type(value).__module__.startswith("textworld.") and callable(
        getattr(value, "serialize", None)
    ):
        return {
            "object_type": f"{type(value).__module__}.{type(value).__name__}",
            "data": plain(value.serialize()),
        }
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return deepcopy(value)


def writer(role):
    """Allocate readable exclusive session names, including across process restarts."""
    root = os.environ["OPD_RECORD_ROOT"]
    # Require explicit base identities instead of inventing revisions from counters.
    for key in ("OPD_STUDENT_REVISION", "OPD_TEACHER_REVISION"):
        if not os.environ.get(key):
            raise ValueError(f"Recording requires {key}")
    while True:
        name = f"{role}-{socket.gethostname()}-{os.getpid()}-session-{next(SESSIONS):06d}"
        try:
            return Recorder(root, name)
        except FileExistsError:
            continue


def episode_trace(task, benchmark):
    """Leave recording inactive when no output root is configured."""
    return EpisodeTrace(task, benchmark) if os.environ.get("OPD_RECORD_ROOT") else nullcontext(None)


class EpisodeTrace:
    """Observe existing calls without reinterpreting rewards or adding model calls."""

    def __init__(self, task, benchmark):
        self.benchmark = benchmark
        self.writer = writer("explorer")
        self.uid = self.writer.path.name + "-episode"
        raw = task.raw_task if isinstance(task.raw_task, dict) else {}
        self.task_id = raw.get("task_id", task.task_desc)
        self.rows = {}
        self.messages = None
        self.phase = "environment_creation"
        self.done = False
        self.raw_reward = None
        self.final_reward = None
        self.scores = {}

    def __enter__(self):
        return self

    def reset(self, observation, info):
        self.observation = plain(observation)
        self.info = plain(info)

    def prompt(self, messages, sampling_kwargs):
        self.messages = plain(messages)
        self.sampling_kwargs = plain(sampling_kwargs)
        self.phase = "student_generation"

    def generated(self, response):
        uid = f"{self.uid}-turn-{len(self.rows) + 1:04d}"
        if response.info is None:
            response.info = {}
        response.info["opd_turn_uid"] = uid
        self.rows[uid] = {
            "turn_uid": uid,
            "episode_uid": self.uid,
            "sampling_student_revision": os.environ["OPD_STUDENT_REVISION"],
            "scoring_teacher_revision": os.environ["OPD_TEACHER_REVISION"],
            "sampling_model_version": plain(response.info.get("model_version")),
            "messages": self.messages,
            "sampling_kwargs": self.sampling_kwargs,
            "tokens": plain(response.tokens),
            "prompt_length": response.prompt_length,
            "response": response.response_text,
            "action": None,
            "observation_before": self.observation,
            "info_before": self.info,
            "observation_after": None,
            "info_after": None,
            "raw_reward": None,
            "episode_done": None,
            "environment_step_executed": None,
            "transition_completed": False,
            "student_logprobs": plain(response.logprobs),
            "teacher_logprobs": None,
            "teacher_scoring_completed": False,
        }

    def action(self, action):
        self.phase = "environment_step"
        self.rows[next(reversed(self.rows))]["action"] = plain(action)

    def step(self, response, action, observation, reward, done, info):
        self.rows[response.info["opd_turn_uid"]].update(
            {
                "action": plain(action),
                "observation_after": plain(observation),
                "info_after": plain(info),
                "raw_reward": plain(reward),
                "episode_done": plain(done),
                "environment_step_executed": True,
                "transition_completed": True,
            }
        )
        self.observation, self.info = plain(observation), plain(info)
        self.done, self.raw_reward = plain(done), plain(reward)

    def action_valid(self, valid):
        self.rows[next(reversed(self.rows))]["environment_step_executed"] = bool(valid)

    def scoring(self, temperature):
        self.phase = "teacher_scoring"
        for row in self.rows.values():
            row["scoring_temperature"] = plain(temperature)

    def scored(self, response):
        row = self.rows[response.info["opd_turn_uid"]]
        row["teacher_logprobs"] = plain(response.teacher_logprobs)
        row["teacher_scoring_completed"] = True

    def finished(self, final_reward, **scores):
        self.final_reward = plain(final_reward)
        self.scores = plain(scores)
        self.phase = "completed"

    def __exit__(self, exc_type, exc, traceback):
        try:
            for row in self.rows.values():
                self.writer.record("turn", **row)
            self.writer.record(
                "episode",
                episode_uid=self.uid,
                task_id=self.task_id,
                benchmark=self.benchmark,
                outcome={
                    **self.scores,
                    "episode_done": self.done,
                    "task_success": None,
                    "raw_score": self.raw_reward,
                    "tcod_final_reward": self.final_reward,
                    "technical_failure": exc_type is not None,
                    "termination_reason": (
                        "exception" if exc_type else "env_terminal" if self.done else "horizon"
                    ),
                    "last_phase": self.phase,
                    "exception_type": exc_type.__name__ if exc_type else None,
                },
            )
        finally:
            self.writer.close()


def update_trace(batch, experiences, step):
    """Wrap only the actual actor update; use final batch IDs after any reordering."""
    if not os.environ.get("OPD_RECORD_ROOT"):
        return nullcontext(None)
    return UpdateTrace(batch, experiences, step)


class UpdateTrace:
    """Record attempts separately from successful and failed actor updates."""

    def __init__(self, batch, experiences, step):
        by_id = {exp.eid.uid: exp for exp in experiences}
        ids = plain(batch.non_tensor_batch["unique_ids"])
        self.data = {
            "update_student_revision": os.environ["OPD_STUDENT_REVISION"],
            "trainer_step_before": step - 1,
            "trainer_step_after_if_completed": step,
            "consumed_turn_uids": [by_id[uid].info["opd_turn_uid"] for uid in ids],
            "experience_uids": ids,
            "mask": plain(batch.batch["response_mask"]),
            "advantage": plain(batch.batch["advantages"]),
            "effective_token_count": int(batch.batch["response_mask"].sum().item()),
            "sampling_model_versions": [plain(by_id[uid].info.get("model_version")) for uid in ids],
        }
        self.writer = writer("trainer")
        self.uid = self.writer.path.name + f"-update-{step}"

    def __enter__(self):
        try:
            self.writer.record("update_attempt", update_attempt_uid=self.uid, **self.data)
        except BaseException:
            self.writer.close()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        try:
            if exc_type is None:
                self.writer.record(
                    "update",
                    update_uid=self.uid,
                    update_attempt_uid=self.uid,
                    status="completed",
                    **self.data,
                )
            else:
                self.writer.record(
                    "update_failure",
                    update_failure_uid=self.uid,
                    update_attempt_uid=self.uid,
                    exception_type=exc_type.__name__,
                    status="failed_or_partially_applied",
                )
        finally:
            self.writer.close()
