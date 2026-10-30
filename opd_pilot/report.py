"""Build study tables, training curves and a source-linked diagnostic casebook."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from opd_pilot.analyze import mean
from opd_pilot.evidence import separate_output
from opd_pilot.paired_stats import task_values


def collect(spec):
    tables = []
    for entry in spec["evaluations"]:
        directory = Path(entry["directory"])
        if not (directory / "episodes.json").is_file() and spec.get("allow_missing", False):
            continue
        episodes = json.loads((directory / "episodes.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "model.json").read_text(encoding="utf-8"))
        actual_step = manifest.get("step", 0)
        if actual_step != entry["step"]:
            raise ValueError("Declared evaluation step differs from model manifest")
        for exclude in (False, True):
            values, counts = task_values(episodes, spec["metric"], exclude)
            tables.append(
                dict(
                    series=entry["series"],
                    step=actual_step,
                    model_revision=manifest["revision"],
                    exclude_technical=exclude,
                    task_equal_score=mean(list(values.values())),
                    known_tasks=len(values),
                    **counts,
                )
            )
    identities = [(row["series"], row["step"], row["exclude_technical"]) for row in tables]
    if len(set(identities)) != len(identities):
        raise ValueError("Combine shards before reporting; duplicate series/step")
    return tables


def load_branches(branch_files):
    """Include contrasting sign patterns and incomplete cases, without success selection."""
    rows = [
        row for path in branch_files for row in json.loads(Path(path).read_text(encoding="utf-8"))
    ]
    # Metric choice remains caller-controlled in the parent report.
    by_prefix = defaultdict(list)
    for row in rows:
        by_prefix[row["prefix_uid"]].append(row)
    return rows, by_prefix


def write_cases(staging, spec):
    from opd_pilot.branch_analysis import summarize

    cases = []
    if spec.get("branch_files"):
        rows, by_prefix = load_branches(spec["branch_files"])
        summary = summarize(rows, spec["metric"], True)
        categories = defaultdict(list)
        for point in summary["points"]:
            a, b = point["delta_teacher"], point["delta_student"]
            category = (
                "incomplete"
                if a is None or b is None
                else f"teacher-{'positive' if a > 0 else 'nonpositive'}_student-{'positive' if b > 0 else 'nonpositive'}"
            )
            categories[category].append(point)
        for depth in range(max(map(len, categories.values()), default=0)):
            for category in sorted(categories):
                if depth >= len(categories[category]) or len(cases) >= spec.get("case_limit", 20):
                    continue
                point = categories[category][depth]
                name = f"case-{len(cases) + 1:03d}"
                branches = by_prefix[point["prefix_uid"]]
                (staging / (name + ".json")).write_text(
                    json.dumps(
                        dict(summary=point, branches=branches), indent=2, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
                text = f"# {name}: {point['prefix_uid']}\n\nCategory: {category}. Metric: {spec['metric']}.\n\n"
                for branch in branches:
                    text += f"## {branch['candidate_owner']} candidate / {branch['executor']} executor / repeat {branch['repeat']}\n\n"
                    text += "Outcome: " + json.dumps(branch["outcome"], ensure_ascii=False) + "\n\n"
                    for turn in branch["turns"]:
                        response = turn.get("response", turn.get("response_text", ""))
                        text += (
                            "Response:\n\n"
                            + "\n".join("> " + line for line in response.splitlines())
                            + "\n\n"
                        )
                        text += (
                            "Observation:\n\n"
                            + "\n".join(
                                "> " + line
                                for line in str(turn.get("observation_after")).splitlines()
                            )
                            + "\n\n"
                        )
                (staging / (name + ".md")).write_text(text, encoding="utf-8")
                cases.append(
                    dict(
                        case=name,
                        category=category,
                        prefix_uid=point["prefix_uid"],
                        task_id=point["task_id"],
                    )
                )
        (staging / "branch-summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    return cases


def build(spec):
    tables = collect(spec)
    if not tables and not spec.get("branch_files") and not spec.get("allow_missing", False):
        raise ValueError("Provide evaluation or branch evidence")
    if type(spec.get("case_limit", 20)) is not int or spec.get("case_limit", 20) < 1:
        raise ValueError("case_limit must be a positive integer")
    sources = [entry["directory"] for entry in spec["evaluations"]] + [
        str(Path(path).parent) for path in spec.get("branch_files", [])
    ]
    output = separate_output(spec["output"], *sources)
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    with (staging / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(tables[0]) if tables else ["series", "step"]
        )
        writer.writeheader()
        writer.writerows(tables)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for series in sorted({row["series"] for row in tables}):
        points = sorted(
            (row for row in tables if row["series"] == series and row["exclude_technical"]),
            key=lambda row: row["step"],
        )
        known = [row for row in points if row["task_equal_score"] is not None]
        axes[0].plot(
            [p["step"] for p in known],
            [p["task_equal_score"] for p in known],
            marker="o",
            label=series,
        )
        axes[1].plot(
            [p["step"] for p in points],
            [p["technical"] / p["submitted"] if p["submitted"] else float("nan") for p in points],
            marker="o",
            label=series,
        )
    axes[0].set(xlabel="Checkpoint step", ylabel=spec["metric"], title="Task-equal known scores")
    axes[1].set(
        xlabel="Checkpoint step",
        ylabel="Technical failure fraction",
        title="All submitted episodes",
    )
    if tables:
        axes[0].legend()
        axes[1].legend()
    fig.tight_layout()
    fig.savefig(staging / "training-and-failures.png", dpi=180)
    plt.close(fig)
    cases = write_cases(staging, spec)
    (staging / "case_index.json").write_text(json.dumps(cases, indent=2), encoding="utf-8")
    (staging / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    missing = [
        entry["directory"]
        for entry in spec["evaluations"]
        if not (Path(entry["directory"]) / "episodes.json").is_file()
    ]
    (staging / "status.json").write_text(
        json.dumps(
            dict(
                status="partial" if missing else "completed",
                missing_evaluations=missing,
                scope="report_generation_only_not_experiment_completion",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    text = "# OPD evidence report\n\nMetric: " + spec["metric"] + ".\n\n"
    text += "Scores use known labels only; inspect metrics.csv for missing labels, technical failures and task coverage.\n\n"
    if missing:
        text += (
            "Missing evaluations (not scored):\n\n"
            + "\n".join("- " + item for item in missing)
            + "\n\n"
        )
    text += "![Training and failures](training-and-failures.png)\n\n"
    text += "Cases cover available sign categories and incomplete branches. They are descriptive, not proof of a training mechanism.\n\n"
    text += "\n".join(
        f"- [{case['case']}](./{case['case']}.md): {case['category']}" for case in cases
    )
    (staging / "report.md").write_text(text + "\n", encoding="utf-8")
    staging.rename(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.validate_only:
        if spec.get("branch_files"):
            from opd_pilot.branch_analysis import summarize

            rows, _ = load_branches(spec["branch_files"])
            summarize(rows, spec["metric"], True)
        print(json.dumps(collect(spec), indent=2))
    else:
        build(spec)


if __name__ == "__main__":
    main()
