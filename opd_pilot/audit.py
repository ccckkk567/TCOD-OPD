"""Audit pilot split membership and recorded model identities offline."""

import argparse
import json
from pathlib import Path

from opd_pilot.evidence import publish, read_records, separate_output
from opd_pilot.prepare import COUNTS


def split_tasks(root):
    """Check the published four-way pilot protocol without opening test games."""
    root = Path(root)
    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    ready = json.loads((root / "SPLITS_READY.json").read_text(encoding="utf-8"))
    if protocol.get("official_test_used") is not False:
        raise ValueError("Pilot protocol must explicitly exclude official test")
    if protocol.get("teacher_success_filter") is not False:
        raise ValueError("Teacher-success filtering is not permitted in this pilot protocol")
    counts = protocol["actual_counts"]
    if set(counts) != set(COUNTS) or ready["actual_counts"] != counts:
        raise ValueError("Split marker or required split names do not match protocol")
    pools = {}
    seen = set()
    for split, count in counts.items():
        rows = [
            json.loads(line)
            for line in (root / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        ids = [row["task_id"] for row in rows]
        if len(rows) != count or len(set(ids)) != len(ids) or seen.intersection(ids):
            raise ValueError("Split count mismatch or overlapping/duplicate task identities")
        if any(row.get("official_split") != "train" for row in rows):
            raise ValueError("Pilot tasks must originate from the official training pool")
        pools[split] = set(ids)
        seen.update(ids)
    return pools


def check(spec):
    """Verify declared split and student/teacher revisions against recorded turns."""
    pools = split_tasks(spec["protocol_root"])
    split = spec["split"]
    if split not in pools:
        raise ValueError(f"Unknown split: {split}")
    records = read_records(spec["records_root"])
    used = {row["task_id"] for row in records["episode"]}
    if used - pools[split]:
        raise ValueError(f"Recorded tasks outside declared split: {sorted(used - pools[split])}")
    for turn in records["turn"]:
        if turn["sampling_student_revision"] != spec["student_revision"]:
            raise ValueError("Recorded sampling student revision differs from specification")
        if turn["scoring_teacher_revision"] != spec["teacher_revision"]:
            raise ValueError("Recorded scoring teacher revision differs from specification")
    return {
        "status": "completed",
        "scope": "recorded_data_membership_and_identity_only",
        "split": split,
        "tasks": len(used),
        "counts": {k: len(v) for k, v in records.items()},
        "checks": [
            "split_disjointness",
            "no_official_test_tasks",
            "record_membership",
            "declared_turn_revisions",
        ],
        "unverified": [
            "backend_token_semantics",
            "actual_model_weight_identity",
            "student_only_execution",
            "environment_success_semantics",
            "checkpoint_recovery",
        ],
    }


def main():
    """Validate or publish an audit; failures exit nonzero without a success artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    output = separate_output(spec["output"], spec["records_root"], spec["protocol_root"])
    result = check(spec)
    if not args.validate_only:
        publish(
            output,
            {"audit.json": result, "status.json": {"status": "completed", "stage": "data_audit"}},
        )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
