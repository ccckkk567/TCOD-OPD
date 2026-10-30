"""Teacher-free, separate-model evaluation using TCOD environment prompts."""

import argparse
import json
import math
from pathlib import Path

from opd_pilot.diagnostic_backend import (
    TransformersBackend,
    model_manifest,
    sample_seed,
)
from opd_pilot.environment import Episode
from opd_pilot.evidence import publish, separate_output
from opd_pilot.integration import plain


def validate(spec):
    """Resolve a frozen task list and explicit evaluation sampling settings."""
    manifest = model_manifest(spec["model_path"])
    if spec["benchmark"] not in ("alfworld", "webshop", "scienceworld"):
        raise ValueError("Unknown benchmark")
    if (
        not math.isfinite(spec["temperature"])
        or spec["temperature"] <= 0
        or not 0 < spec["top_p"] <= 1
        or spec["top_k"] != 0
    ):
        raise ValueError("Require positive temperature, top_p in (0,1], top_k=0")
    for key in ("max_new_tokens", "max_env_steps", "repeats"):
        if type(spec[key]) is not int or spec[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    tasks = [
        json.loads(line)
        for line in Path(spec["tasks"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not tasks or len({task["task_id"] for task in tasks}) != len(tasks):
        raise ValueError("Require nonempty unique task IDs")
    for task in tasks:
        if "task_desc" not in task and "game_file" not in task:
            raise ValueError("Each task requires task_desc (or ALFWorld game_file)")
    return manifest, tasks


def rollout(spec, task, repeat, backend, factory=Episode):
    """Keep technical failures in the denominator; never infer success from done."""
    rows, env = [], None
    failure = None
    outcome = dict(task_success=None, technical_failure=True, termination_reason="exception")
    try:
        env = factory(
            spec["benchmark"], task.get("task_desc", task.get("game_file")), spec["max_env_steps"]
        )
        for step in range(spec["max_env_steps"]):
            messages = env.prompt()
            prefix = backend.encode_messages(messages, spec.get("enable_thinking"))
            seed = sample_seed(spec["seed"], task["task_id"], repeat, step)
            response = backend.generate(prefix, spec, seed)
            row = dict(messages=messages, prefix_tokens=prefix, seed=seed, **response)
            rows.append(row)
            row.update(env.apply(response["response_text"]))
            if env.done:
                break
        outcome = env.outcome()
    except Exception as error:
        failure = error
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                if failure is None:
                    failure = error
    if failure is not None:
        outcome.update(
            technical_failure=True,
            termination_reason="exception",
            exception_type=type(failure).__name__,
        )
    return dict(task_id=task["task_id"], repeat=repeat, turns=rows, outcome=outcome)


def write_records(directory, episodes, manifest, benchmark):
    """Expose teacher-free evaluation trajectories to the existing probe/audit tools."""
    from opd_pilot.records import Recorder

    with Recorder(directory / "records", "evaluation") as writer:
        for index, episode in enumerate(episodes, 1):
            episode_uid = f"evaluation-{index:06d}"
            writer.record(
                "episode",
                episode_uid=episode_uid,
                task_id=episode["task_id"],
                benchmark=benchmark,
                outcome=plain(episode["outcome"]),
            )
            for turn_index, turn in enumerate(episode["turns"], 1):
                prefix, response = turn["prefix_tokens"], turn["response_tokens"]
                writer.record(
                    "turn",
                    turn_uid=f"{episode_uid}-turn-{turn_index:04d}",
                    episode_uid=episode_uid,
                    sampling_student_revision=manifest["revision"],
                    scoring_teacher_revision=None,
                    messages=turn["messages"],
                    tokens=prefix + response,
                    prompt_length=len(prefix),
                    response=turn["response_text"],
                    action=turn.get("action"),
                    raw_reward=plain(turn.get("raw_reward")),
                    student_logprobs=None,
                    teacher_logprobs=None,
                    teacher_scoring_completed=False,
                    observation_before=plain(turn.get("observation_before")),
                    info_before=plain(turn.get("info_before")),
                    observation_after=plain(turn.get("observation_after")),
                    info_after=plain(turn.get("info_after")),
                    episode_done=plain(turn.get("episode_done")),
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], spec["model_path"], Path(spec["tasks"]).parent)
    manifest, tasks = validate(spec)
    if args.validate_only:
        print(json.dumps(dict(status="spec_validated", tasks=len(tasks))))
        return
    backend = TransformersBackend(spec["model_path"])
    episodes = [
        rollout(spec, task, repeat, backend) for task in tasks for repeat in range(spec["repeats"])
    ]
    episodes = plain(episodes)
    failures = sum(row["outcome"]["technical_failure"] for row in episodes)
    publish(
        output,
        {
            "episodes.json": episodes,
            "spec.json": spec,
            "model.json": manifest,
            "status.json": dict(
                status="completed_with_failures" if failures else "completed",
                episodes=len(episodes),
                technical_failures=failures,
                tcod_backend_equivalence_validated=False,
            ),
        },
        extra_writer=lambda directory: write_records(
            directory, episodes, manifest, spec["benchmark"]
        ),
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
