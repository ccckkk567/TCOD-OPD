"""Single-GPU diagnostic interaction and scoring; not native OPD training."""

import argparse
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from opd_pilot.diagnostic_backend import TransformersBackend, model_manifest, token_identity
from opd_pilot.environment import Episode
from orion.scripts.environment_interface_smoke import json_value


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=json_value), encoding="utf-8")


def interact(benchmark, tasks, helpers, backend):
    episodes = []
    settings = dict(temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=512)
    for index, task in enumerate(tasks):
        episode = Episode(benchmark, task, max_steps=3, utilities=helpers)
        turns = []
        try:
            for step in range(3):
                messages = episode.prompt()
                prefix = backend.encode_messages(messages, enable_thinking=False)
                response = backend.generate(prefix, settings, seed=42 + index * 3 + step)
                if not response["response_tokens"]:
                    raise RuntimeError("Student generated no tokens")
                transition = episode.apply(response["response_text"])
                turns.append(
                    dict(messages=messages, prefix_tokens=prefix, transition=transition, **response)
                )
                print(f"task={index + 1} turn={step + 1} action={transition['action']}", flush=True)
                if episode.done:
                    break
            episodes.append(dict(task=task, turns=turns, outcome=episode.outcome()))
        finally:
            episode.close()
    return dict(settings=settings, enable_thinking=False, episodes=episodes)


def score_turns(episodes, backend):
    count = 0
    for episode in episodes:
        for turn in episode["turns"]:
            scores = backend.score(turn["prefix_tokens"], turn["response_tokens"], 1.0)
            if len(scores) != len(turn["response_tokens"]):
                raise RuntimeError("Teacher scores do not align with student tokens")
            turn["teacher_logprobs"] = scores
            count += 1
    if not count:
        raise RuntimeError("No student turns to score")
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("alfworld", "scienceworld"))
    parser.add_argument("--phase", choices=("student", "teacher"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--models-root", type=Path, required=True)
    parser.add_argument("--alfworld-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    if args.phase == "student":
        args.output.mkdir(parents=True, exist_ok=False)
    try:
        student = args.models_root / "Qwen3-1.7B"
        teacher = args.models_root / "Qwen3-8B"
        manifests = dict(student=model_manifest(student), teacher=model_manifest(teacher))
        if token_identity(student) != token_identity(teacher):
            raise ValueError("Student and teacher token-ID mappings differ")
        if args.phase == "student":
            helpers = SimpleNamespace(
                **runpy.run_path(
                    str(repo / "trinity/common/workflows/envs/TCOD" / args.benchmark / "utils.py")
                )
            )
            if args.benchmark == "alfworld":
                rows = [
                    json.loads(line)
                    for line in args.alfworld_tasks.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                tasks = [row["game_file"] for row in rows[:2]]
                if len(tasks) != 2 or not all(Path(task).is_file() for task in tasks):
                    raise ValueError("Need two existing ALFWorld smoke tasks")
            else:
                jar = args.data_root / "scienceworld/scienceworld.jar"
                if not jar.is_file():
                    raise FileNotFoundError(jar)
                tasks = [
                    dict(task_name=name, var_num=0, jar_path=str(jar)) for name in ("boil", "melt")
                ]
            result = interact(args.benchmark, tasks, helpers, TransformersBackend(student))
            result.update(
                benchmark=args.benchmark,
                models=manifests,
                code_revision=(repo / "revision.txt").read_text().strip(),
            )
            write_json(args.output / "student.json", result)
            print("STUDENT_INTERACTION_COMPLETED", flush=True)
        else:
            result = json.loads((args.output / "student.json").read_text(encoding="utf-8"))
            if result["benchmark"] != args.benchmark or result["models"] != manifests:
                raise ValueError("Student artifact benchmark/model identity mismatch")
            count = score_turns(result["episodes"], TransformersBackend(teacher))
            write_json(args.output / "scored.json", result)
            status = dict(
                status="diagnostic_interaction_and_scoring_passed",
                benchmark=args.benchmark,
                episodes=len(result["episodes"]),
                scored_turns=count,
                backend="transformers",
                scoring_context="student_prefix",
                model_interaction_validated=True,
                teacher_token_scoring_validated=True,
                tcod_teacher_context_validated=False,
                production_workflow_validated=False,
                training_validated=False,
                recovery_validated=False,
                task_success_claimed=False,
            )
            write_json(args.output / "status.json", status)
            print(json.dumps(status), flush=True)
    except Exception as error:
        write_json(
            args.output / "failure.json",
            dict(
                status="failed",
                phase=args.phase,
                error_type=type(error).__name__,
                message=str(error),
            ),
        )
        raise


if __name__ == "__main__":
    main()
