"""Exercise existing TCOD simulator helpers, independently of training/model sampling."""

import argparse
import importlib.metadata
import json
import runpy
from pathlib import Path


def scienceworld(jar, helpers):
    results = []
    for task in ("boil", "melt"):
        config = dict(task_name=task, var_num=0, jar_path=str(jar))
        env = helpers["_create_scienceworld_env"](
            config, max_env_steps=30, generate_gold_path=False
        )
        try:
            observation, info, description = helpers["_reset_scienceworld_env"](env)
            actions, objects = helpers["_get_compact_action_info"](env)
            after, reward, done, after_info = env.step("look around")
            results.append(
                dict(
                    task=config,
                    description=description,
                    reset_observation=observation,
                    reset_info=info,
                    action_templates=actions,
                    objects=objects,
                    step_action="look around",
                    observation_after=after,
                    reward=reward,
                    done=done,
                    info_after=after_info,
                )
            )
        finally:
            env.close()
    return results


def webshop(helpers):
    env = helpers["_create_webshop_env"]()
    try:
        env.reset(session=0)
        observation = env.observation
        actions = env.get_available_actions()
        after, reward, done, info = env.step("search[shirt]")
        return [
            dict(
                session=0,
                reset_observation=observation,
                available_actions=actions,
                step_action="search[shirt]",
                observation_after=after,
                reward=reward,
                done=done,
                info_after=info,
            )
        ]
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("benchmark", choices=("scienceworld", "webshop"))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    helpers = runpy.run_path(
        str(repo / "trinity/common/workflows/envs/TCOD" / args.benchmark / "utils.py")
    )
    try:
        rows = (
            scienceworld(args.data_root / "scienceworld.jar", helpers)
            if args.benchmark == "scienceworld"
            else webshop(helpers)
        )
        result = dict(
            status="reset_step_passed",
            benchmark=args.benchmark,
            episodes=rows,
            training_validated=False,
            task_success_claimed=False,
        )
        if args.benchmark == "scienceworld":
            result["version"] = importlib.metadata.version("scienceworld")
        else:
            result["data_asset"] = json.loads(
                (args.data_root / "asset.json").read_text(encoding="utf-8")
            )
        (args.output / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({key: value for key, value in result.items() if key != "episodes"}))
    except Exception as error:
        (args.output / "failure.json").write_text(
            json.dumps(
                dict(status="failed", exception_type=type(error).__name__, message=str(error)),
                indent=2,
            ),
            encoding="utf-8",
        )
        raise


if __name__ == "__main__":
    main()
