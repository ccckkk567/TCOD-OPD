"""Shared offline evidence IO; no TCOD or model imports."""

import json
from pathlib import Path

from opd_pilot.records import audit


def read_records(root):
    """Read only completed, linked records, preserving the actual stored payloads."""
    audit(root)
    result = {kind: [] for kind in ("episode", "turn", "update")}
    for path in sorted(Path(root).glob("*/events.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                item = json.loads(line)
                result[item["kind"]].append(item["evidence"])
    return result


def publish(output, payloads):
    """Publish a new artifact directory, retaining failed staging for inspection."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    for name, value in payloads.items():
        (staging / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)


def separate_output(output, *sources):
    """Keep outputs outside input evidence trees and refuse replacement."""
    output = Path(output).resolve()
    for source in sources:
        source = Path(source).resolve()
        if source == output or source in output.parents or output in source.parents:
            raise ValueError("Output must be separate from input evidence trees")
    if output.exists():
        raise FileExistsError(output)
    return output
