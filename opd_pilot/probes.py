"""Select a fixed random prefix pool from actual student development records."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from opd_pilot.audit import check
from opd_pilot.evidence import publish, read_records, separate_output


def select(spec):
    """Sample without success filtering; retain exact messages and prefix token IDs."""
    if spec["split"] != "analysis_dev":
        raise ValueError("Exploratory fixed probes must use analysis_dev, not train/confirmation")
    check(spec)
    limit, per_task = spec["max_prefixes"], spec["max_per_task"]
    if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (limit, per_task)):
        raise ValueError("Prefix limits must be positive integers")
    records = read_records(spec["records_root"])
    episodes = {row["episode_uid"]: row for row in records["episode"]}
    groups = defaultdict(list)
    for row in sorted(records["turn"], key=lambda row: row["turn_uid"]):
        if not isinstance(row["tokens"], list) or any(
            isinstance(token, bool) or not isinstance(token, int) or token < 0
            for token in row["tokens"]
        ):
            raise ValueError("Recorded tokens must be a list of nonnegative integer IDs")
        if not isinstance(row["messages"], list) or not row["messages"]:
            raise ValueError("Actual recorded chat messages are required")
        length = row["prompt_length"]
        if (
            isinstance(length, bool)
            or not isinstance(length, int)
            or not 0 < length <= len(row["tokens"])
        ):
            raise ValueError("Invalid recorded prompt length")
        task = episodes[row["episode_uid"]]["task_id"]
        groups[task].append(row)
    rng = random.Random(spec["seed"])
    tasks = sorted(groups)
    rng.shuffle(tasks)
    for task in tasks:
        rng.shuffle(groups[task])
    selected = []
    for depth in range(per_task):
        for task in tasks:
            if depth < len(groups[task]) and len(selected) < limit:
                row = groups[task][depth]
                selected.append(
                    {
                        "prefix_uid": f"prefix-{len(selected) + 1:04d}",
                        "task_id": task,
                        "source_turn_uid": row["turn_uid"],
                        "episode_uid": row["episode_uid"],
                        "messages": row["messages"],
                        "prefix_tokens": row["tokens"][: row["prompt_length"]],
                        "sampling_student_revision": row["sampling_student_revision"],
                        "selection_pool": "random_main",
                        "environment_replay_validated": False,
                    }
                )
    if not selected:
        raise ValueError("No recorded development prefixes available")
    return {
        "schema_version": 1,
        "seed": spec["seed"],
        "requested": limit,
        "actual": len(selected),
        "max_per_task": per_task,
        "selection": "seeded_task_balanced_random_without_success_filter",
        "candidate_generation_complete": False,
        "prefixes": selected,
    }


def main():
    """Publish an immutable pool; validation mode never writes output."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], spec["records_root"], spec["protocol_root"])
    result = select(spec)
    if not args.validate_only:
        publish(
            output,
            {
                "prefix_bank.json": result,
                "spec.json": spec,
                "status.json": {"status": "completed", "stage": "prefix_selection_only"},
            },
        )
    print(json.dumps({key: value for key, value in result.items() if key != "prefixes"}, indent=2))


if __name__ == "__main__":
    main()
