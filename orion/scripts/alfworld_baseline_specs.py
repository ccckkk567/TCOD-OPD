"""Freeze a common 32-task development baseline; never select on model outcomes."""

import argparse
import json
import random
from pathlib import Path

from opd_pilot.prepare import validate_splits


def prepare(protocol, output, models):
    validate_splits(protocol)
    rows = [json.loads(line) for line in (protocol / "analysis_dev.jsonl").read_text().splitlines()]
    if len(rows) < 32:
        raise ValueError("Need 32 distinct development tasks")
    rows = sorted(rows, key=lambda row: row["task_id"])
    random.Random(42).shuffle(rows)
    output.mkdir(parents=True, exist_ok=False)
    tasks = output / "tasks.jsonl"
    tasks.write_text("".join(json.dumps(row) + "\n" for row in rows[:32]), encoding="utf-8")
    for key, name in (("s17", "Qwen3-1.7B"), ("t08", "Qwen3-8B")):
        spec = dict(
            benchmark="alfworld",
            model_path=str(models / name),
            tasks=str(tasks),
            output=str(output.parent / (output.name + "-" + key)),
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=512,
            max_env_steps=30,
            repeats=2,
            seed=42,
            enable_thinking=False,
        )
        (output / f"{key}.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.protocol, args.output, args.models)
