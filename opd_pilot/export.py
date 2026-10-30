"""Export a closed FSDP checkpoint with the pinned veRL merger and validate HF files."""

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

from opd_pilot.assets import validate_model
from opd_pilot.checkpoints import inspect
from opd_pilot.evidence import separate_output


def validate(spec):
    checkpoint = Path(spec["checkpoint"])
    if not re.fullmatch(r"global_step_[0-9]+", checkpoint.name):
        raise ValueError("Use a complete global_step_N checkpoint directory")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", spec["run_name"]):
        raise ValueError("Use a readable run identity")
    report = inspect(checkpoint, spec["world_size"])
    if report["status"] != "structure_complete":
        raise ValueError("Checkpoint structure is incomplete")
    if not (checkpoint / "actor/huggingface/config.json").is_file():
        raise ValueError("Checkpoint lacks the HF config required by veRL's merger")
    return report


def export(spec):
    """Publish READY only after merger success and loading every safe-tensor shard."""
    from safetensors import safe_open

    report = validate(spec)
    if importlib.metadata.version("verl") != "0.7.0":
        raise ValueError("This export adapter was checked against veRL 0.7.0")
    source = Path(spec["checkpoint"]).resolve()
    output = separate_output(spec["output"], source)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    before = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (source / "actor").glob("*.pt")
    }
    subprocess.run(
        [
            sys.executable,
            "-m",
            "verl.model_merger",
            "merge",
            "--backend",
            "fsdp",
            "--local_dir",
            str(source / "actor"),
            "--target_dir",
            str(staging),
        ],
        check=True,
    )
    after = {
        path.name: (path.stat().st_size, path.stat().st_mtime_ns)
        for path in (source / "actor").glob("*.pt")
    }
    if after != before:
        raise ValueError("Checkpoint changed during export; wait for writers to stop")
    files = {
        path.relative_to(staging).as_posix(): path.stat().st_size
        for path in staging.rglob("*")
        if path.is_file()
    }
    step = int(source.name.removeprefix("global_step_"))
    manifest = dict(
        repo_id="local-checkpoint/" + spec["run_name"],
        revision=f"{spec['run_name']}-step-{step}",
        files=files,
        source_checkpoint=str(source),
        step=step,
        world_size=spec["world_size"],
        kind="checkpoint_export",
        recovery_validated=False,
        merger_version="0.7.0",
    )
    validate_model(staging, manifest)
    tensor_keys = set()
    for path in sorted(staging.glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for key in reader.keys():
                if key in tensor_keys:
                    raise ValueError("Duplicate exported tensor key")
                tensor_keys.add(key)
                # Materialize one tensor at a time to catch unreadable payloads without a second full model.
                tensor = reader.get_tensor(key)
                del tensor
    index = staging / "model.safetensors.index.json"
    if not tensor_keys:
        raise ValueError("No exported model tensors")
    if index.exists() and set(json.loads(index.read_text())["weight_map"]) != tensor_keys:
        raise ValueError("Exported tensor keys differ from the HF index")
    (staging / "checkpoint_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for name in ("asset.json", "READY.json"):
        (staging / name).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    staging.rename(output)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = validate(spec) if args.validate_only else export(spec)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
