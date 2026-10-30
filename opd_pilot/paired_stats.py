"""Task-level paired differences and task-cluster bootstrap from episode evidence."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from opd_pilot.analyze import mean, number
from opd_pilot.evidence import publish, read_records, separate_output


def task_values(episodes, metric, exclude_technical):
    """Aggregate repeated episodes within tasks before any resampling."""
    values = defaultdict(list)
    counts = {
        "submitted": len(episodes),
        "technical": 0,
        "technical_unknown": 0,
        "excluded_technical": 0,
        "missing_metric": 0,
        "used": 0,
    }
    for episode in episodes:
        outcome = episode["outcome"]
        technical = outcome.get("technical_failure")
        if technical is not None and not isinstance(technical, bool):
            raise ValueError("technical_failure must be true, false or null")
        counts["technical"] += int(technical is True)
        counts["technical_unknown"] += int(technical is None)
        if exclude_technical and technical is True:
            counts["excluded_technical"] += 1
            continue
        value = outcome.get(metric)
        if metric == "task_success":
            if value is not None and not isinstance(value, bool):
                raise ValueError("task_success must be true, false or null")
            value = int(value) if value is not None else None
        else:
            value = number(value)
        if value is None:
            counts["missing_metric"] += 1
        else:
            values[episode["task_id"]].append(value)
            counts["used"] += 1
    return {key: mean(items) for key, items in values.items()}, counts


def percentile(values, probability):
    """Linear interpolation between adjacent ordered bootstrap estimates."""
    position = (len(values) - 1) * probability
    low = int(position)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (position - low)


def compare(left, right, metric, seed=42, replicates=2000, exclude_technical=False):
    """Compare right minus left, pairing tasks, not turns or individual seeds."""
    if metric not in ("task_success", "score_final", "score_best", "raw_score"):
        raise ValueError("Unsupported episode metric")
    if isinstance(replicates, bool) or not isinstance(replicates, int) or replicates < 2:
        raise ValueError("At least two bootstrap replicates required")
    a, counts_a = task_values(left, metric, exclude_technical)
    b, counts_b = task_values(right, metric, exclude_technical)
    paired = sorted(a.keys() & b.keys())
    rows = [
        {"task_id": key, "left": a[key], "right": b[key], "difference": b[key] - a[key]}
        for key in paired
    ]
    differences = [row["difference"] for row in rows]
    interval = None
    if len(paired) >= 2:
        rng = random.Random(seed)
        estimates = sorted(
            mean(rng.choices(differences, k=len(differences))) for _ in range(replicates)
        )
        interval = [percentile(estimates, 0.025), percentile(estimates, 0.975)]
    return {
        "metric": metric,
        "direction": "right_minus_left",
        "paired_tasks": len(paired),
        "left_only_tasks": sorted(a.keys() - b.keys()),
        "right_only_tasks": sorted(b.keys() - a.keys()),
        "mean_difference": mean(differences),
        "bootstrap_percentile_95": interval,
        "bootstrap_unit": "task",
        "bootstrap_replicates": replicates,
        "seed": seed,
        "left_counts": counts_a,
        "right_counts": counts_b,
        "per_task": rows,
        "technical_policy": "exclude_explicit_true_only" if exclude_technical else "retain_all",
        "notes": [
            "Missing labels are never filled with zero.",
            "Tasks need a known metric on both sides; inspect unmatched tasks and counts.",
            "A single paired task has no bootstrap interval; small pools require caution.",
            "No significance or causal conclusion is automatically asserted.",
        ],
    }


def calculate(spec):
    """Return all-episode and explicit-technical-failure-excluded results together."""
    left = read_records(spec["left_records"])["episode"]
    right = read_records(spec["right_records"])["episode"]
    kwargs = {key: spec[key] for key in ("metric", "seed", "replicates")}
    return {
        "all_submitted_known_metrics": compare(left, right, **kwargs),
        "excluding_explicit_technical_failures": compare(
            left, right, exclude_technical=True, **kwargs
        ),
        "context": spec["context"],
    }


def main():
    """Analyze immutable paired evidence specified by a JSON file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], spec["left_records"], spec["right_records"])
    result = calculate(spec)
    if not args.validate_only:
        publish(
            output,
            {
                "paired_statistics.json": result,
                "spec.json": spec,
                "status.json": {"status": "completed", "stage": "paired_statistics"},
            },
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
