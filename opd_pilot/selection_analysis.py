"""Compare fixed-response likelihood and continuation changes across checkpoints."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from opd_pilot.analyze import mean, number
from opd_pilot.branch_analysis import summarize
from opd_pilot.evidence import separate_output


def likelihoods(entry, spec, candidates):
    groups, scored = defaultdict(list), set()
    for path in entry["scores"]:
        bank = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            bank["scoring_revision"] != entry["revision"]
            or bank["temperature"] != spec["temperature"]
        ):
            raise ValueError("Scoring revision/temperature differs from comparison protocol")
        for score in bank["scores"]:
            uid = score["candidate_uid"]
            if uid in scored:
                raise ValueError("Duplicate candidate score")
            scored.add(uid)
            candidate = candidates[uid]
            if (
                score["response_tokens"] != candidate["response_tokens"]
                or score["prefix_uid"] != candidate["prefix_uid"]
            ):
                raise ValueError("Scored response differs from fixed candidate")
            value = number(score["mean_token_nll"])
            if value is None:
                raise ValueError("Missing/nonfinite response likelihood")
            groups[(candidate["prefix_uid"], candidate["owner"])].append(value)
    if scored != candidates.keys():
        raise ValueError("Every checkpoint must score the same complete candidate bank")
    return groups


def adoption(entry, candidates):
    """Optional fresh sampling measures exact parsed-action frequency, not response NLL."""
    fixed, sampled = defaultdict(list), defaultdict(list)
    for candidate in candidates.values():
        fixed[candidate["prefix_uid"]].append(candidate)
    for path in entry.get("generated_banks", []):
        bank = json.loads(Path(path).read_text(encoding="utf-8"))
        for row in bank["candidates"]:
            if row["generation_revision"] != entry["revision"] or row["owner"] != "student":
                raise ValueError(
                    "Adoption frequency requires the declared checkpoint's student samples"
                )
            reference = fixed[row["prefix_uid"]][0]
            if row["prefix_tokens"] != reference["prefix_tokens"]:
                raise ValueError("Adoption samples use a different prefix")
            sampled[row["prefix_uid"]].append(row["parsed_action"])
    result = {}
    for prefix, actions in sampled.items():
        owners = {
            owner: {row["parsed_action"] for row in fixed[prefix] if row["owner"] == owner}
            for owner in ("student", "teacher")
        }
        result[prefix] = dict(
            adoption_samples=len(actions),
            teacher_action_frequency=mean([int(action in owners["teacher"]) for action in actions]),
            student_action_frequency=mean([int(action in owners["student"]) for action in actions]),
            overlapping_action_frequency=mean(
                [int(action in owners["teacher"] & owners["student"]) for action in actions]
            ),
        )
    return result


def calculate(spec):
    candidates = {}
    for path in spec["candidate_banks"]:
        for row in json.loads(Path(path).read_text(encoding="utf-8"))["candidates"]:
            uid = row["candidate_uid"]
            if uid in candidates:
                raise ValueError("Duplicate candidate identity")
            candidates[uid] = row
    results, identities = [], set()
    for entry in spec["checkpoints"]:
        identity = (entry["series"], entry["step"])
        if identity in identities:
            raise ValueError("Duplicate checkpoint series/step")
        identities.add(identity)
        groups = likelihoods(entry, spec, candidates)
        frequencies = adoption(entry, candidates)
        branches = [
            row
            for path in entry["branch_files"]
            for row in json.loads(Path(path).read_text(encoding="utf-8"))
        ]
        for branch in branches:
            candidate = candidates[branch["candidate_uid"]]
            expected = (
                entry["revision"] if branch["executor"] == "student" else spec["teacher_revision"]
            )
            if branch.get("executor_revision") != expected:
                raise ValueError("Continuation executor revision differs from checkpoint protocol")
            if (
                candidate["prefix_uid"] != branch["prefix_uid"]
                or candidate["owner"] != branch["candidate_owner"]
                or candidate["task_id"] != branch["task_id"]
            ):
                raise ValueError("Continuation differs from fixed candidate identity")
        points = summarize(branches, spec["metric"], True)["points"]
        for point in points:
            prefix = point["prefix_uid"]
            student, teacher = (mean(groups[(prefix, owner)]) for owner in ("student", "teacher"))
            results.append(
                dict(
                    series=entry["series"],
                    step=entry["step"],
                    revision=entry["revision"],
                    prefix_uid=prefix,
                    task_id=point["task_id"],
                    teacher_minus_student_mean_token_nll=None
                    if student is None or teacher is None
                    else teacher - student,
                    continuation_delta_student=point["delta_student"],
                    continuation_delta_teacher=point["delta_teacher"],
                    **frequencies.get(
                        prefix,
                        dict(
                            adoption_samples=0,
                            teacher_action_frequency=None,
                            student_action_frequency=None,
                            overlapping_action_frequency=None,
                        ),
                    ),
                )
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    rows = calculate(spec)
    if args.validate_only:
        print(json.dumps(dict(comparison_rows=len(rows))))
        return
    sources = list(spec["candidate_banks"])
    for entry in spec["checkpoints"]:
        sources.extend(entry["scores"] + entry["branch_files"] + entry.get("generated_banks", []))
    output = separate_output(spec["output"], *[Path(path).parent for path in sources])
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    with (staging / "prefix_changes.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else ["series", "step"])
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    keys = (
        "teacher_minus_student_mean_token_nll",
        "teacher_action_frequency",
        "continuation_delta_student",
    )
    for series in sorted({row["series"] for row in rows}):
        for axis, key in zip(axes, keys):
            points = []
            for step in sorted({row["step"] for row in rows if row["series"] == series}):
                tasks = defaultdict(list)
                for row in rows:
                    if row["series"] == series and row["step"] == step and row[key] is not None:
                        tasks[row["task_id"]].append(row[key])
                value = mean([mean(values) for values in tasks.values()])
                if value is not None:
                    points.append((step, value))
            axis.plot([p[0] for p in points], [p[1] for p in points], marker="o", label=series)
    axes[0].set(ylabel="Teacher minus student response mean-token NLL", xlabel="Checkpoint step")
    axes[1].set(ylabel="Exact teacher-candidate action frequency", xlabel="Checkpoint step")
    axes[2].set(ylabel="Student continuation value difference", xlabel="Checkpoint step")
    for axis in axes:
        if rows:
            axis.legend()
    fig.tight_layout()
    fig.savefig(staging / "likelihood-and-continuation.png", dpi=180)
    plt.close(fig)
    (staging / "spec.json").write_text(json.dumps(spec, indent=2), encoding="utf-8")
    (staging / "status.json").write_text(
        json.dumps(
            dict(
                status="completed",
                rows=len(rows),
                scope="fixed_full_response_likelihood_not_action_probability_or_causal_learning",
                branch_revision_scope="recorded_executor_asset_revision",
            ),
            indent=2,
        ),
        encoding="utf-8",
    )
    staging.rename(output)


if __name__ == "__main__":
    main()
