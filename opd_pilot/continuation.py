"""Replay recorded states, inject whole candidates, and run diagnostic continuations."""

import argparse
import json
from pathlib import Path

from opd_pilot.diagnostic_backend import TransformersBackend, sample_seed
from opd_pilot.environment import Episode
from opd_pilot.evidence import publish, read_records, separate_output
from opd_pilot.integration import plain


class ReplayMismatch(ValueError):
    """The observed state/history did not match the recorded prefix."""


def restore(env, history, target):
    """Replay complete assistant responses, checking prompts and exposed transitions."""
    for row in history:
        if env.prompt() != row["messages"]:
            raise ReplayMismatch("Historical prompt mismatch")
        transition = plain(env.apply(row["response"]))
        for key in (
            "action",
            "observation_before",
            "observation_after",
            "raw_reward",
            "episode_done",
        ):
            if transition[key] != row[key]:
                raise ReplayMismatch(f"Historical {key} mismatch")
        if env.done:
            raise ReplayMismatch("Replay terminated before target")
    messages = env.prompt()
    if messages != target["messages"] or plain(env.observation) != target["observation_before"]:
        raise ReplayMismatch("Target state/prompt mismatch")
    # Exposed state only: this is not proof of equality of unobservable simulator state.
    if plain(env.info) != target["info_before"]:
        raise ReplayMismatch("Target environment info mismatch")
    return messages


def branch(spec, candidate, history, target, task, repeat, backend, factory=Episode):
    env, rows, replayed = None, [], False
    outcome = dict(task_success=None, technical_failure=True, termination_reason="exception")
    failure = None
    try:
        env = factory(
            spec["benchmark"],
            task.get("task_desc", task.get("game_file")),
            spec["max_env_steps"],
            **({"context_mode": spec["context_mode"]} if "context_mode" in spec else {}),
        )
        restore(env, history, target)
        replayed = True
        rows.append(
            dict(response=candidate["response_text"], **env.apply(candidate["response_text"]))
        )
        for step in range(len(history) + 1, spec["max_env_steps"]):
            if env.done:
                break
            messages = env.prompt()
            prefix = backend.encode_messages(messages, spec.get("enable_thinking"))
            seed = sample_seed(spec["seed"], candidate["candidate_uid"], repeat, step)
            response = backend.generate(prefix, spec, seed)
            row = dict(messages=messages, prefix_tokens=prefix, seed=seed, **response)
            rows.append(row)
            row.update(env.apply(response["response_text"]))
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
            termination_reason=(
                "replay_mismatch" if isinstance(failure, ReplayMismatch) else "exception"
            ),
            exception_type=type(failure).__name__,
        )
    return dict(
        candidate_uid=candidate["candidate_uid"],
        prefix_uid=candidate["prefix_uid"],
        task_id=candidate["task_id"],
        candidate_owner=candidate["owner"],
        executor=spec["executor"],
        repeat=repeat,
        replay_validated=replayed,
        replay_scope="exposed_observations_info_and_prompt",
        turns=rows,
        outcome=outcome,
    )


def inputs(spec):
    from opd_pilot.evaluate import validate

    manifest, tasks = validate(spec)
    if spec["executor"] not in ("student", "teacher"):
        raise ValueError("Executor must be student or teacher")
    roots = {
        str(Path(path).resolve())
        for path in [spec["records_root"], *spec.get("additional_records_roots", [])]
    }
    cache = {}
    tasks = {row["task_id"]: row for row in tasks}
    bank = json.loads(Path(spec["candidate_bank"]).read_text(encoding="utf-8"))
    candidates = bank["candidates"]
    if not candidates or len({row["candidate_uid"] for row in candidates}) != len(candidates):
        raise ValueError("Require unique candidates")
    work = []
    for candidate in candidates:
        root = str(Path(candidate.get("records_root") or spec["records_root"]).resolve())
        if root not in roots:
            raise ValueError("Declare every common-pool records root explicitly in the spec")
        if root not in cache:
            records = read_records(root)
            cache[root] = (
                {row["turn_uid"]: row for row in records["turn"]},
                {row["episode_uid"]: row for row in records["episode"]},
            )
        turns, episodes = cache[root]
        target = turns[candidate["source_turn_uid"]]
        episode = episodes[target["episode_uid"]]
        if candidate["prefix_tokens"] != target["tokens"][: target["prompt_length"]]:
            raise ValueError("Candidate prefix tokens differ from recorded prefix")
        if (
            episode["task_id"] != candidate["task_id"]
            or candidate["messages"] != target["messages"]
        ):
            raise ValueError("Candidate source does not match recorded task/prompt")
        if episode.get("benchmark", spec["benchmark"]) != spec["benchmark"]:
            raise ValueError("Benchmark mismatch")
        # Integration IDs carry fixed-width chronological turn numbers.
        episode_turns = sorted(
            (row for row in turns.values() if row["episode_uid"] == target["episode_uid"]),
            key=lambda row: row["turn_uid"],
        )
        index = next(
            i for i, row in enumerate(episode_turns) if row["turn_uid"] == target["turn_uid"]
        )
        if index >= spec["max_env_steps"]:
            raise ValueError("Candidate falls outside the episode horizon")
        work.append((candidate, episode_turns[:index], target, tasks[candidate["task_id"]]))
    return manifest, work


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(
        spec["output"],
        spec["records_root"],
        spec["model_path"],
        Path(spec["candidate_bank"]).parent,
        *spec.get("additional_records_roots", []),
    )
    manifest, work = inputs(spec)
    if args.validate_only:
        print(json.dumps(dict(status="spec_validated", branches=len(work) * spec["repeats"])))
        return
    backend = TransformersBackend(spec["model_path"])
    branches = [
        branch(spec, *item, repeat, backend) for item in work for repeat in range(spec["repeats"])
    ]
    for row in branches:
        row["executor_revision"] = manifest["revision"]
    branches = plain(branches)
    failures = sum(row["outcome"]["technical_failure"] for row in branches)
    publish(
        output,
        {
            "branches.json": branches,
            "spec.json": spec,
            "model.json": manifest,
            "status.json": dict(
                status="completed_with_failures" if failures else "completed",
                branches=len(branches),
                technical_failures=failures,
            ),
        },
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
