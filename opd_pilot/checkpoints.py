"""Inspect TCOD FSDP checkpoint structure without loading pickled tensors."""

import argparse
import json
import zipfile
from pathlib import Path

from opd_pilot.evidence import publish, separate_output


def inspect(checkpoint, world_size):
    """Require model, optimizer and extra-state archives for every declared rank.

    This catches absent, empty or truncated archive structures. It does not prove
    tensor integrity, scheduler/RNG contents, completed asynchronous writes or
    successful restoration. Inspect only after the checkpoint writers stop.
    """
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size < 1:
        raise ValueError("world_size must be a positive integer")
    actor = Path(checkpoint) / "actor"
    missing, invalid, files = [], [], []
    expected = {
        f"{kind}_world_size_{world_size}_rank_{rank}.pt"
        for kind in ("model", "optim", "extra_state")
        for rank in range(world_size)
    }
    for name in sorted(expected):
        path = actor / name
        if not path.is_file():
            missing.append(name)
            continue
        try:
            with zipfile.ZipFile(path) as archive:
                names = archive.namelist()
                if not any(item.endswith("/data.pkl") for item in names):
                    raise ValueError("Missing PyTorch serialized object")
            files.append({"name": name, "bytes": path.stat().st_size})
        except (OSError, ValueError, zipfile.BadZipFile):
            invalid.append(name)
    unexpected = sorted(
        path.name for path in actor.glob("*_world_size_*_rank_*.pt") if path.name not in expected
    )
    return {
        "status": "structure_complete" if not (missing or invalid or unexpected) else "incomplete",
        "scope": "fsdp_archive_structure_only",
        "checkpoint": str(Path(checkpoint).resolve()),
        "world_size": world_size,
        "missing": missing,
        "invalid": invalid,
        "unexpected": unexpected,
        "files": files,
        "recovery_validated": False,
        "pending": [
            "tensor_payload_integrity",
            "optimizer_scheduler_rng_restore",
            "data_cursor_restore",
            "consumed_sample_ids",
            "real_continuation_update",
        ],
    }


def main():
    """Inspect a real checkpoint; never export incomplete shards as a ready model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], spec["checkpoint"])
    result = inspect(spec["checkpoint"], spec["world_size"])
    if not args.validate_only:
        publish(
            output,
            {
                "checkpoint_report.json": result,
                "status.json": {"status": result["status"], "stage": "checkpoint_structure"},
            },
        )
    print(json.dumps(result, indent=2))
    if result["status"] != "structure_complete":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
