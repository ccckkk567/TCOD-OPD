"""Render the user's approved smoke and pilot settings on the original OPD layout."""

import argparse
import json
from pathlib import Path

from opd_pilot.configs import REMOTE, STORAGE, TEMPLATE, differences, render
from opd_pilot.evidence import separate_output


def configure(spec, template):
    config = render(spec, template)
    phase = spec["phase"]
    if phase not in ("smoke-first", "smoke-resume", "pilot"):
        raise ValueError("Unknown approved phase")
    target = {"smoke-first": 2, "smoke-resume": 4, "pilot": 50}[phase]
    config["continue_from_checkpoint"] = phase == "smoke-resume"
    config["buffer"]["total_steps"] = target
    config["trainer"].update(
        total_steps=target,
        schedule_steps=100,
        save_interval=0,
        save_steps=[10, 25, 50] if phase == "pilot" else [2, 4],
    )
    config["monitor"] = dict(
        monitor_type="swanlab",
        monitor_args=dict(
            mode="online", logdir=REMOTE + f"/logs/tcod-opd/{spec['run_name']}/swanlab"
        ),
    )
    root = STORAGE + f"/outputs/tcod-opd/{spec['run_name']}/{phase}"
    # Revisions are resolved from actual asset manifests by the runtime driver.
    config["explorer"]["env_vars"].update(
        OPD_RECORD_ROOT=root + "/records",
        OPD_STOP_FILE=root + "/STOP_REQUESTED",
        OPD_RUN_STATUS=root + "/training-status.json",
    )
    return config


def main():
    import yaml

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    config = configure(spec, template)
    changes = differences(template, config)
    if args.validate_only:
        print(json.dumps(changes, indent=2))
        return
    output = separate_output(spec["output"], TEMPLATE.parent)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (staging / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (staging / "changes.json").write_text(json.dumps(changes, indent=2), encoding="utf-8")
    (staging / "status.json").write_text(
        json.dumps(
            dict(
                status="rendered_not_runtime_validated",
                phase=spec["phase"],
                gpu_count=8,
                target_steps=config["trainer"]["total_steps"],
                schedule_steps=100,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    staging.rename(output)


if __name__ == "__main__":
    main()
