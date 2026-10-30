"""Real simulator/prompt/parser checks; no model sampling, teacher or optimizer."""

import argparse
import importlib.metadata
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

from opd_pilot.environment import Episode


def run_episode(benchmark, task, helpers):
    episode = Episode(benchmark, task, max_steps=4, utilities=helpers)
    turns = []
    try:
        for index in range(3):
            messages = episode.prompt()
            if len(messages) != 2 * index + 1:
                raise RuntimeError("Full response history was not preserved")
            if benchmark == "alfworld":
                actions = episode.info.get("admissible_commands", [])
                if actions and isinstance(actions[0], list):
                    actions = actions[0]
                if "look" not in actions:
                    raise RuntimeError("ALFWorld does not expose the smoke action 'look'")
                action = "look"
            else:
                action = "look around"
            response = f"<think>Inspect the environment.</think><action>{action}</action>"
            transition = episode.apply(response)
            if transition["action"] != action or not transition["environment_step_executed"]:
                raise RuntimeError("Original parser/action interface failed")
            if not isinstance(episode.observation, str) or not episode.observation.strip():
                raise RuntimeError("Simulator returned an empty/non-text observation")
            turns.append(dict(messages=messages, response=response, transition=transition))
            if episode.done:
                raise RuntimeError("Episode ended before the three-turn history check completed")
        return dict(task=task, task_description=episode.task_description, turns=turns)
    finally:
        episode.close()


def json_value(value):
    # Simulator info can contain NumPy arrays/scalars. Preserve values, not repr strings.
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Unsupported simulator metadata: {type(value).__name__}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("alfworld", "scienceworld"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--alfworld-tasks", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    try:
        helpers = SimpleNamespace(
            **runpy.run_path(
                str(repo / "trinity/common/workflows/envs/TCOD" / args.benchmark / "utils.py")
            )
        )
        if args.benchmark == "alfworld":
            rows = [json.loads(line) for line in args.alfworld_tasks.read_text().splitlines()]
            tasks = [row["game_file"] for row in rows[:2]]
            if len(tasks) != 2 or not all(Path(task).is_file() for task in tasks):
                raise ValueError("Need two existing tasks from the reserved ALFWorld smoke split")
        else:
            jar = args.data_root / "scienceworld/scienceworld.jar"
            if not jar.is_file():
                raise FileNotFoundError(jar)
            tasks = [
                dict(task_name=task, var_num=0, jar_path=str(jar)) for task in ("boil", "melt")
            ]
        result = dict(
            status="environment_interface_passed",
            benchmark=args.benchmark,
            version=importlib.metadata.version(args.benchmark),
            code_revision=(repo / "revision.txt").read_text().strip(),
            action_source="scripted_smoke_only",
            scope="original_helpers_and_diagnostic_episode_three_turns",
            model_interaction_validated=False,
            production_workflow_validated=False,
            training_validated=False,
            task_success_claimed=False,
            episodes=[run_episode(args.benchmark, task, helpers) for task in tasks],
        )
        (args.output / "result.json").write_text(
            json.dumps(result, indent=2, default=json_value), encoding="utf-8"
        )
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(dict(status="failed", error_type=type(error).__name__, message=str(error))),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
