"""Summarize recorded outcomes offline without inferring environment success."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from opd_pilot.records import audit


def mean(values):
    """Return null rather than a fabricated zero for an empty denominator."""
    return sum(values) / len(values) if values else None


def number(value):
    """Accept finite numeric scores only; booleans are not scores."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite score or null, got {value!r}")
    return value


def summarize(root):
    """Report all recorded episodes, including unknown and technical outcomes.

    Call once per environment/split/model/protocol. No failures are dropped and no
    cross-environment score normalization or teacher-quality gate is applied.
    """
    evidence = audit(root)
    episodes = []
    turn_counts = Counter()
    for path in sorted(Path(root).glob("*/events.jsonl")):
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                item = record["evidence"]
                if record["kind"] == "episode":
                    episodes.append(item)
                elif record["kind"] == "turn":
                    turn_counts[item["episode_uid"]] += 1
    return _aggregate(episodes, turn_counts, evidence)


def _label(value, positive, negative):
    if value is None:
        return "unknown"
    if not isinstance(value, bool):
        raise ValueError("Outcome flags must be true, false or null")
    return positive if value else negative


def _aggregate(episodes, turn_counts, evidence):
    success = Counter()
    technical = Counter()
    termination = Counter()
    scores = {"score_final": [], "score_best": [], "raw_score": []}
    task_results = defaultdict(list)
    steps = []
    successful_steps = []
    for episode in episodes:
        outcome = episode["outcome"]
        if not isinstance(outcome, dict):
            raise ValueError("Episode outcome must be an object")
        result = outcome.get("task_success")
        failure = outcome.get("technical_failure")
        success[_label(result, "success", "failure")] += 1
        technical[_label(failure, "yes", "no")] += 1
        reason = outcome.get("termination_reason")
        if reason is not None and not isinstance(reason, str):
            raise ValueError("termination_reason must be a string or null")
        termination[reason if reason is not None else "unknown"] += 1
        if result is not None:
            task_results[episode["task_id"]].append(int(result))
        for key in scores:
            value = number(outcome.get(key))
            if value is not None:
                scores[key].append(value)
        count = turn_counts[episode["episode_uid"]]
        steps.append(count)
        if result is True:
            successful_steps.append(count)
    known = success["success"] + success["failure"]
    total = len(episodes)
    return {
        "schema_version": 1,
        "evidence": evidence,
        "episodes": total,
        "unique_tasks": len({item["task_id"] for item in episodes}),
        "success": {key: success[key] for key in ("success", "failure", "unknown")},
        "success_rate_known": success["success"] / known if known else None,
        "success_label_coverage": known / total if total else None,
        "task_equal_success_rate_known": mean([mean(v) for v in task_results.values()]),
        "tasks_with_known_success": len(task_results),
        "technical_failure": {key: technical[key] for key in ("yes", "no", "unknown")},
        "termination_reasons": dict(sorted(termination.items())),
        "scores": {
            key: {"mean": mean(values), "count": len(values)} for key, values in scores.items()
        },
        "recorded_turns_mean": mean(steps),
        "successful_recorded_turns_mean": mean(successful_steps),
        "notes": [
            "Success is read only from task_success; done and reward do not determine it.",
            "Success rate uses known labels only; inspect coverage and unknown count.",
            "Technical failures are retained; no automatic technical-failure filtering.",
            "Score keys are separate; raw_score is not substituted for score_final.",
            "Turn counts describe recorded turns, not verified simulator step counts.",
            "One environment/split/model/protocol per input; comparability is caller responsibility.",
        ],
    }


def validate_spec(spec):
    """Validate the offline report specification without creating output."""
    required = ("records_root", "output", "environment", "split", "model_revision", "protocol")
    if any(not isinstance(spec.get(key), str) or not spec[key].strip() for key in required):
        raise ValueError(f"Required nonempty string fields: {required}")
    source, output = Path(spec["records_root"]).resolve(), Path(spec["output"]).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Report output must be separate from the source record directory")
    if output.exists():
        raise FileExistsError(output)
    return source, output


def run(spec):
    """Publish a separate report directory only after successful analysis."""
    source, output = validate_spec(spec)
    summary = summarize(source)
    summary["context"] = {
        key: spec[key] for key in ("environment", "split", "model_revision", "protocol")
    }
    staging = output.with_name(output.name + ".in-progress")
    staging.mkdir(parents=True, exist_ok=False)
    (staging / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Recorded outcome report",
        "",
        f"Episodes: {summary['episodes']}",
        f"Success / failure / unknown: {summary['success']}",
        f"Success rate among known labels: {summary['success_rate_known']}",
        f"Success-label coverage: {summary['success_label_coverage']}",
        "",
    ]
    lines.extend(f"- {note}" for note in summary["notes"])
    (staging / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (staging / "status.json").write_text(
        json.dumps({"status": "completed", "stage": "offline_analysis"}) + "\n", encoding="utf-8"
    )
    if output.exists():
        raise FileExistsError(output)
    staging.rename(output)
    return summary


def main():
    """Analyze a spec; validation mode reads evidence and writes nothing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.validate_only:
        source, _ = validate_spec(spec)
        print(json.dumps(summarize(source), indent=2))
    else:
        print(json.dumps(run(spec), indent=2))


if __name__ == "__main__":
    main()
