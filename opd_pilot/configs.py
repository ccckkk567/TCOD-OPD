"""Render Orion paths into the original ALFWorld OPD configuration and validate it."""

import argparse
import copy
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

from opd_pilot.assets import MODELS
from opd_pilot.evidence import separate_output

REMOTE = "/scratch/prjsnnn0000269/Students/shuming/cjk"
STORAGE = "/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk"
TEMPLATE = Path(__file__).resolve().parents[1] / "TCOD_examples/alfworld/opd.yaml"


def render(spec, template):
    """Change only explicitly selected model/data paths and operational run identity."""
    run = spec["run_name"]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", run):
        raise ValueError("Use a readable run name without path separators")
    student, teacher = spec["student"], spec["teacher"]
    if student not in ("s17", "s04") or teacher not in ("t08", "t30"):
        raise ValueError("Unsupported pilot student/teacher identity")
    tasks = PurePosixPath(spec["train_tasks"])
    if not tasks.is_absolute() or not str(tasks).startswith(STORAGE + "/outputs/tcod-opd/"):
        raise ValueError("Use the published pilot task list under storage_ext outputs/tcod-opd")
    if tasks.name != "train_pilot.jsonl":
        raise ValueError("Training input must be train_pilot.jsonl")
    config = copy.deepcopy(template)
    config["project"] = "tcod-opd"
    config["name"] = run
    config["checkpoint_root_dir"] = STORAGE + "/checkpoints"
    config["ray_namespace"] = "tcod-opd-" + run
    config["model"]["model_path"] = REMOTE + "/models/" + MODELS[student].split("/")[1]
    config["explorer"]["auxiliary_models"][0]["model_path"] = (
        REMOTE + "/models/" + MODELS[teacher].split("/")[1]
    )
    config["buffer"]["explorer_input"]["taskset"]["path"] = str(tasks)
    config["buffer"]["trainer_input"]["experience_buffer"]["path"] = (
        "sqlite:///" + STORAGE + f"/outputs/tcod-opd/{run}/experience.db"
    )
    config["explorer"]["env_vars"] = {
        **config["explorer"].get("env_vars", {}),
        "TMPDIR": REMOTE + f"/tmp/{run}/workers",
        "RAY_TMPDIR": REMOTE + f"/tmp/{run}/ray",
    }
    return config


def differences(before, after, path=""):
    """List every rendered leaf change so behavior changes cannot hide in a template."""
    if isinstance(before, dict) and isinstance(after, dict):
        rows = []
        for key in sorted(before.keys() | after.keys()):
            rows.extend(differences(before.get(key), after.get(key), f"{path}.{key}".strip(".")))
        return rows
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [
            row
            for index, (a, b) in enumerate(zip(before, after))
            for row in differences(a, b, f"{path}.{index}")
        ]
    return [] if before == after else [{"field": path, "before": before, "after": after}]


def build(spec, validate_only=False):
    """Render locally; runtime validation is a separate real-TCOD operation."""
    import yaml

    output = separate_output(spec["output"], TEMPLATE.parent)
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    config = render(spec, template)
    changes = differences(template, config)
    if validate_only:
        return {"status": "render_validated_only", "changes": changes}
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=TEMPLATE.parents[2], text=True
    ).strip()
    metadata = {
        "status": "rendered_not_runtime_validated",
        "source_revision": revision,
        "template": "TCOD_examples/alfworld/opd.yaml",
        "spec": spec,
        "changes": changes,
        "gpu_count": config["cluster"]["gpu_per_node"],
        "buffer_steps": config["buffer"]["total_steps"],
        "trainer_steps": config["trainer"]["total_steps"],
        "notes": [
            "Original algorithm, sampling, reward, batch sizes and steps preserved.",
            "This is not yet the proposed 50-step pilot or authorization to train.",
        ],
    }
    (staging / "status.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return metadata


def runtime_check(config_path, output):
    """Call TCOD's actual loader and validator in an isolated validation directory."""
    import yaml

    from trinity.common.config import load_config
    from trinity.utils.plugin_loader import load_plugins

    output = separate_output(output, Path(config_path).parent)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    load_plugins()
    config = load_config(str(config_path))
    # The original validator writes checkpoint directories. Keep those writes in this
    # validation artifact, never in the real experiment checkpoint directory.
    config.checkpoint_root_dir = str(staging / "validator-work")
    config.check_and_update()
    config.save(str(staging / "validated-preview.yaml"))
    cluster = config.cluster
    result = {
        "status": "runtime_config_validated",
        "training_ready": False,
        "gpu_mapping": {
            "total": cluster.total_gpu_num,
            "rollout": cluster.rollout_gpu_num,
            "teacher": cluster.auxiliary_model_gpu_num,
            "trainer": cluster.trainer_gpu_num,
        },
        "note": "validated-preview.yaml uses isolated scratch paths; never launch training from it.",
    }
    (staging / "status.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    # Ensure the serialized validator result remains a readable YAML document.
    yaml.safe_load((staging / "validated-preview.yaml").read_text(encoding="utf-8"))
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return result


def main():
    """Generate with --spec, or run actual TCOD validation using --runtime-config."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--spec", type=Path)
    source.add_argument("--runtime-config", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    if args.spec:
        result = build(json.loads(args.spec.read_text(encoding="utf-8")), args.validate_only)
    else:
        if args.output is None or args.validate_only:
            parser.error("Runtime check requires --output and does not accept --validate-only")
        result = runtime_check(args.runtime_config, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
