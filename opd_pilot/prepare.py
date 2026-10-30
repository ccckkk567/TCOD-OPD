"""Prepare disjoint ALFWorld pilot lists from real, unfiltered training games."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from opd_pilot.assets import write_json

COUNTS = {"train_pilot": 256, "analysis_dev": 64, "confirm_holdout": 32, "smoke": 4}


def allocate(counts, total):
    """Proportional allocation using largest remainders and stable tie breaking."""
    available = sum(counts.values())
    if available == 0:
        return dict.fromkeys(counts, 0)
    total = min(total, available)
    result = {key: value * total // available for key, value in counts.items()}
    ranked = sorted(counts, key=lambda key: (-(counts[key] * total % available), key))
    for key in ranked[: total - sum(result.values())]:
        result[key] += 1
    return result


def validate_splits(output):
    """Reject missing games, duplicated tasks and incomplete split publications."""
    output = Path(output)
    protocol = json.loads((output / "protocol.json").read_text(encoding="utf-8"))
    seen = set()
    for name, count in protocol["actual_counts"].items():
        rows = [json.loads(line) for line in (output / f"{name}.jsonl").read_text().splitlines()]
        if len(rows) != count:
            raise ValueError(f"Split count mismatch: {name}")
        for row in rows:
            if row["task_id"] in seen or not Path(row["game_file"]).is_file():
                raise ValueError("Duplicated task or missing game file")
            seen.add(row["task_id"])
    return protocol


def prepare_alfworld(data_root, output, seed=42, counts=None):
    """Reserve development/confirmation/smoke games without reading official test pools."""
    data_root, output = Path(data_root).resolve(), Path(output)
    counts = dict(COUNTS if counts is None else counts)
    if any(value < 0 for value in counts.values()) or sum(counts.values()) == 0:
        raise ValueError("Split sizes must be nonnegative with at least one requested game")
    train_root = data_root / "json_2.1.1/train"
    games = sorted(train_root.glob("*/*/game.tw-pddl"))
    if not games:
        raise ValueError(f"No real ALFWorld training games found under {train_root}")
    if output.exists() and any(output.iterdir()):
        raise ValueError(
            "Output already contains a protocol; use it or choose a new readable version"
        )
    groups = defaultdict(list)
    for game in games:
        relative = game.relative_to(train_root)
        category = relative.parts[0].split("-")[0]
        groups[category].append(game)
    rng = random.Random(seed)
    for category in sorted(groups):
        rng.shuffle(groups[category])
    actual = counts if sum(counts.values()) <= len(games) else allocate(counts, len(games))
    output.mkdir(parents=True, exist_ok=True)
    categories = {}
    for name, count in actual.items():
        quotas = allocate({key: len(value) for key, value in groups.items()}, count)
        rows = []
        for category in sorted(groups):
            for _ in range(quotas[category]):
                game = groups[category].pop()
                rows.append(
                    {
                        "game_file": str(game),
                        "target": "",
                        "task_id": game.relative_to(train_root).as_posix(),
                        "category": category,
                        "official_split": "train",
                    }
                )
        rng.shuffle(rows)
        with (output / f"{name}.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        categories[name] = quotas
    protocol = {
        "benchmark": "alfworld",
        "data_root": str(data_root),
        "seed": seed,
        "requested_counts": counts,
        "actual_counts": actual,
        "category_counts": categories,
        "available_training_games": len(games),
        "official_test_used": False,
        "teacher_success_filter": False,
        "selection": "proportional_by_task_category",
        "environment_version": None,
        "environment_smoke_passed": False,
        "note": "Task lists only; simulator version and behavior protocol are not yet validated.",
    }
    write_json(output / "protocol.json", protocol)
    validate_splits(output)
    write_json(output / "SPLITS_READY.json", {"actual_counts": actual, "seed": seed})
    return protocol


def main():
    """Generate pilot task lists in a CPU allocation once benchmark assets are available."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    protocol = (
        validate_splits(args.output)
        if args.validate_only
        else prepare_alfworld(args.data_root, args.output, args.seed)
    )
    print(json.dumps(protocol, indent=2))


if __name__ == "__main__":
    main()
