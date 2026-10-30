"""Summarize fixed-candidate 2x2 continuations with explicit missing denominators."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from opd_pilot.analyze import mean, number
from opd_pilot.evidence import separate_output


def summarize(rows, metric, exclude_failures):
    if metric not in ("task_success", "raw_score", "score_final", "score_best"):
        raise ValueError("Choose an explicit supported outcome metric")
    identities, groups, tasks, counts = set(), defaultdict(list), {}, defaultdict(int)
    candidate_sets = defaultdict(set)
    for row in rows:
        prefix, owner, executor, candidate = (
            row[key] for key in ("prefix_uid", "candidate_owner", "executor", "candidate_uid")
        )
        if owner not in ("student", "teacher") or executor not in ("student", "teacher"):
            raise ValueError("Unknown candidate owner/executor")
        identity = (prefix, candidate, executor, row["repeat"])
        if identity in identities:
            raise ValueError("Duplicate branch; do not combine overlapping shards")
        identities.add(identity)
        if prefix in tasks and tasks[prefix] != row["task_id"]:
            raise ValueError("Prefix identity spans different tasks")
        tasks[prefix] = row["task_id"]
        counts["submitted"] += 1
        failed = row["outcome"].get("technical_failure") is True
        counts["technical_failures"] += int(failed)
        counts["replay_unverified"] += int(row.get("replay_validated") is not True)
        if row.get("replay_validated") is not True or (exclude_failures and failed):
            continue
        value = row["outcome"].get(metric)
        if metric == "task_success":
            if value is not None and type(value) is not bool:
                raise ValueError("Success must be explicitly boolean or unknown")
            value = int(value) if value is not None else None
        else:
            value = number(value)
        if value is None:
            counts["missing_metric"] += 1
            continue
        groups[(prefix, owner, executor, candidate)].append(value)
        candidate_sets[(prefix, owner, executor)].add(candidate)
        counts["known_metric"] += 1
    points = []
    for prefix in sorted(tasks):
        point = dict(prefix_uid=prefix, task_id=tasks[prefix])
        for executor in ("student", "teacher"):
            cells = {}
            for owner in ("student", "teacher"):
                # Restrict each candidate to evidence from both executor policies.
                common = (
                    candidate_sets[(prefix, owner, "student")]
                    & candidate_sets[(prefix, owner, "teacher")]
                )
                values = [
                    mean(groups[(prefix, owner, executor, candidate)])
                    for candidate in sorted(common)
                ]
                cells[owner] = mean(values)
                point[f"{owner}_candidate_{executor}_executor"] = cells[owner]
                point[f"{owner}_candidate_paired_count"] = len(common)
            point[f"delta_{executor}"] = (
                None if None in cells.values() else cells["teacher"] - cells["student"]
            )
        points.append(point)
    complete = [
        point
        for point in points
        if point["delta_student"] is not None and point["delta_teacher"] is not None
    ]
    task_deltas = defaultdict(list)
    for point in complete:
        task_deltas[point["task_id"]].append(point)
    return dict(
        metric=metric,
        exclude_technical_failures=exclude_failures,
        counts=dict(counts),
        prefixes=len(points),
        complete_prefixes=len(complete),
        incomplete_prefixes=len(points) - len(complete),
        exploratory_reversal_fraction=mean(
            [int(p["delta_teacher"] > 0 and p["delta_student"] < 0) for p in complete]
        ),
        task_equal_delta_student=mean(
            [mean([p["delta_student"] for p in values]) for values in task_deltas.values()]
        ),
        task_equal_delta_teacher=mean(
            [mean([p["delta_teacher"] for p in values]) for values in task_deltas.values()]
        ),
        points=points,
        claim_scope="descriptive_fixed_candidate_continuation_not_training_causality",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], *[Path(path).parent for path in spec["branch_files"]])
    rows = [
        row
        for path in spec["branch_files"]
        for row in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
    results = {
        "all_known_metrics": summarize(rows, spec["metric"], False),
        "excluding_technical_failures": summarize(rows, spec["metric"], True),
    }
    if args.validate_only:
        print(json.dumps({key: value["counts"] for key, value in results.items()}))
        return
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    points = results["excluding_technical_failures"]["points"]
    if points:
        with (staging / "prefix_results.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(points[0]))
            writer.writeheader()
            writer.writerows(points)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    valid = [
        point
        for point in points
        if point["delta_student"] is not None and point["delta_teacher"] is not None
    ]
    ax.scatter([p["delta_teacher"] for p in valid], [p["delta_student"] for p in valid])
    ax.axhline(0, color="gray", linewidth=0.7)
    ax.axvline(0, color="gray", linewidth=0.7)
    ax.set(
        xlabel="Teacher executor: teacher candidate minus student candidate",
        ylabel="Student executor: teacher candidate minus student candidate",
        title=f"{spec['metric']}: {len(valid)}/{len(points)} complete prefixes",
    )
    fig.tight_layout()
    fig.savefig(staging / "continuation-deltas.png", dpi=180)
    plt.close(fig)
    (staging / "case_index.json").write_text(json.dumps(points, indent=2), encoding="utf-8")
    (staging / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (staging / "status.json").write_text(
        json.dumps(
            dict(
                status="completed",
                stage="descriptive_branch_analysis",
                complete_prefixes=len(valid),
                incomplete_prefixes=len(points) - len(valid),
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    staging.rename(output)


if __name__ == "__main__":
    main()
