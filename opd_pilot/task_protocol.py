"""Freeze disjoint pilot splits from an explicit benchmark training-task manifest."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from opd_pilot.evidence import separate_output
from opd_pilot.prepare import COUNTS


def split(spec):
    rows = [
        json.loads(line)
        for line in Path(spec["tasks"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    counts = spec["counts"]
    if set(counts) != set(COUNTS) or any(
        type(value) is not int or value < 1 for value in counts.values()
    ):
        raise ValueError("Specify positive counts for all four protocol splits")
    if len({row["task_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate task identity")
    if any(row.get("official_split") != "train" for row in rows):
        raise ValueError("Every input must explicitly belong to the official training pool")
    if any("task_desc" not in row and "game_file" not in row for row in rows):
        raise ValueError("Each task needs its concrete simulator description")
    if sum(counts.values()) > len(rows):
        raise ValueError("Insufficient distinct training tasks")
    rng = random.Random(spec["seed"])
    groups = defaultdict(list)
    for row in sorted(rows, key=lambda row: row["task_id"]):
        groups[row.get("stratum", "all")].append(row)
    names = sorted(groups)
    rng.shuffle(names)
    for group in groups.values():
        rng.shuffle(group)
    ordered = [
        groups[name][depth]
        for depth in range(max(map(len, groups.values())))
        for name in names
        if depth < len(groups[name])
    ]
    result, position = {}, 0
    for name in COUNTS:
        result[name] = ordered[position : position + counts[name]]
        position += counts[name]
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = split(spec)
    metadata = dict(
        actual_counts={name: len(rows) for name, rows in result.items()},
        official_test_used=False,
        teacher_success_filter=False,
        benchmark=spec["benchmark"],
        seed=spec["seed"],
        source_manifest=str(Path(spec["tasks"]).resolve()),
        identity_scope="input_manifest_training_labels_not_simulator_verified",
    )
    if args.validate_only:
        print(json.dumps(metadata, indent=2))
        return
    output = separate_output(spec["output"], Path(spec["tasks"]).parent)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    for name, rows in result.items():
        (staging / (name + ".jsonl")).write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
        )
    for name in ("protocol.json", "SPLITS_READY.json"):
        (staging / name).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (staging / "status.json").write_text(
        json.dumps(dict(status="completed", stage="task_split_only", simulator_verified=False)),
        encoding="utf-8",
    )
    staging.rename(output)


if __name__ == "__main__":
    main()
