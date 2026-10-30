"""Offline E5 bundles and branch planning. No optimizer/replay execution entry point."""

import argparse
import hashlib
import json
import math
from pathlib import Path

from opd_pilot.checkpoints import inspect
from opd_pilot.evidence import publish, separate_output


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def require(value, names):
    for name in names:
        if name not in value or value[name] is None or value[name] == "":
            raise ValueError(f"Missing {name}")


def validate_batch(batch):
    require(batch, ("student_revision", "checkpoint_revision", "sampling", "rows"))
    if batch["purpose"] != "e5_fresh_student_train" or batch["official_split"] != "train":
        raise ValueError(
            "E5 requires fresh student training data, not diagnostic/holdout trajectories"
        )
    if batch["student_revision"] != batch["checkpoint_revision"]:
        raise ValueError("Student revision differs from source checkpoint")
    if not batch["rows"]:
        raise ValueError("Empty batch")
    seen = set()
    for row in batch["rows"]:
        require(row, ("sample_id", "tokens", "prompt_length", "response_mask", "native_fields"))
        if row["sample_id"] in seen:
            raise ValueError("Duplicate sample ID")
        seen.add(row["sample_id"])
        tokens, start = row["tokens"], row["prompt_length"]
        if any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("Token IDs must be nonnegative integers")
        if type(start) is not int or not 0 < start < len(tokens):
            raise ValueError("Invalid prompt boundary")
        mask = row["response_mask"]
        if len(mask) != len(tokens) - start or any(x not in (0, 1) for x in mask):
            raise ValueError("Invalid response mask; no automatic intersection or truncation")
        if not any(mask):
            raise ValueError("No effective response tokens")
        # Native payload stays opaque: conversion back to Experience requires the later adapter.
        if not isinstance(row["native_fields"], dict):
            raise ValueError("native_fields must retain the source field mapping")
    digest(batch)


def alignment(batch, scores):
    validate_batch(batch)
    require(scores, ("teacher_revision", "tokenizer_revision", "backend", "temperature", "rows"))
    issues = []
    if scores.get("batch_digest") != digest(batch):
        issues.append("batch_digest")
    if len(scores["rows"]) != len(batch["rows"]):
        issues.append("row_count")
    for index, (sample, scored) in enumerate(zip(batch["rows"], scores["rows"])):
        for field in ("sample_id", "tokens", "prompt_length", "response_mask"):
            if sample[field] != scored.get(field):
                issues.append(f"rows[{index}].{field}")
        values = scored.get("teacher_log_probs", [])
        mask = sample["response_mask"]
        if len(values) != len(mask):
            issues.append(f"rows[{index}].score_length")
        elif any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
            issues.append(f"rows[{index}].nonfinite_score")
        if scored.get("valid_mask") != mask:
            issues.append(f"rows[{index}].valid_mask")
    return dict(aligned=not issues, differences=issues, automatic_exclusions=False)


def freeze(kind, payload, output, batch=None):
    if kind == "batch":
        validate_batch(payload)
    elif kind == "score":
        report = alignment(batch, payload)
        if not report["aligned"]:
            raise ValueError(json.dumps(report))
    else:
        raise ValueError("Unknown bundle type")
    publish(
        output,
        {
            "payload.json": payload,
            "manifest.json": dict(
                kind=kind,
                schema_version=1,
                digest=digest(payload),
                executable=False,
                native_roundtrip_validated=False,
            ),
        },
    )


def load_bundle(path, kind):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    payload = json.loads((path / "payload.json").read_text(encoding="utf-8"))
    if (
        manifest["schema_version"] != 1
        or manifest["kind"] != kind
        or manifest["digest"] != digest(payload)
    ):
        raise ValueError("Bundle type/version/integrity mismatch")
    return payload


def plan(spec):
    """Audit source structure and score alignment; never load pickle or create branches."""
    require(spec, ("checkpoint", "world_size", "batch", "scores", "branches_root"))
    batch = load_bundle(spec["batch"], "batch")
    validate_batch(batch)
    sources = [spec["checkpoint"], spec["batch"], *spec["scores"].values()]
    root = separate_output(spec["branches_root"], *sources)
    report = inspect(spec["checkpoint"], spec["world_size"])
    # Hash checkpoint files only on explicit invocation, after writers have stopped.
    for item in report["files"]:
        item["sha256"] = file_digest(Path(spec["checkpoint"]) / "actor" / item["name"])
    scores = {name: load_bundle(path, "score") for name, path in spec["scores"].items()}
    if set(scores) != {"A", "B"}:
        raise ValueError("Declare exactly A and B; scoring conditions must be explicit")
    checks = {name: alignment(batch, score) for name, score in scores.items()}
    temperature_match = scores["A"]["temperature"] == scores["B"]["temperature"]
    return dict(
        status="offline_audit_only",
        checkpoint=report,
        score_alignment=checks,
        temperature_match=temperature_match,
        structurally_eligible=(
            report["status"] == "structure_complete"
            and all(x["aligned"] for x in checks.values())
            and temperature_match
        ),
        branches={name: str(root / name) for name in ("A", "B", "AA-repeat")},
        no_update_reference=spec["checkpoint"],
        source_batch_digest=digest(batch),
        scoring_metadata={
            name: {key: value for key, value in score.items() if key != "rows"}
            for name, score in scores.items()
        },
        execution_authorized=False,
        native_executor_implemented=False,
        unresolved=[
            "native Experience roundtrip",
            "complete optimizer/scheduler/RNG restore",
            "actual logical optimizer step count",
            "native versus replay equivalence",
            "A/A noise",
        ],
    )


def compare(spec):
    """Compare explicitly named, protocol-matched scalar summaries; missing stays null."""
    rows = {}
    for name in ("C", "A", "B"):
        path = spec["results"].get(name)
        rows[name] = json.loads(Path(path).read_text(encoding="utf-8")) if path else None
        if rows[name] is not None and rows[name]["protocol_digest"] != spec["protocol_digest"]:
            raise ValueError("Result protocols differ")
    result = {}
    for metric in spec["metrics"]:
        values = {
            name: row.get("metrics", {}).get(metric) if row else None for name, row in rows.items()
        }
        if any(
            value is not None and (type(value) not in (int, float) or not math.isfinite(value))
            for value in values.values()
        ):
            raise ValueError("Metrics must be finite numbers or null")
        for label, left, right in (("A-C", "A", "C"), ("B-C", "B", "C"), ("B-A", "B", "A")):
            values[label] = (
                None
                if values[left] is None or values[right] is None
                else values[left] - values[right]
            )
        result[metric] = values
    return dict(
        status="descriptive_comparison",
        metrics=result,
        statistical_claim=None,
        missing=[name for name, row in rows.items() if row is None],
        update_evidence={
            name: row.get("update_evidence") if row else None for name, row in rows.items()
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "operation", choices=("freeze-batch", "freeze-score", "align", "plan", "compare")
    )
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if args.operation.startswith("freeze-"):
        payload = json.loads(Path(spec["input"]).read_text(encoding="utf-8"))
        batch = load_bundle(spec["batch"], "batch") if args.operation == "freeze-score" else None
        output = separate_output(spec["output"], spec["input"], *([spec["batch"]] if batch else []))
        freeze(args.operation[7:], payload, output, batch)
        return
    if args.operation == "align":
        result = alignment(load_bundle(spec["batch"], "batch"), load_bundle(spec["score"], "score"))
    else:
        result = plan(spec) if args.operation == "plan" else compare(spec)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result.get("aligned") is False or result.get("structurally_eligible") is False:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
