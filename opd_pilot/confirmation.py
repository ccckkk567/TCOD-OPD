"""Freeze an E6 confirmation protocol without submitting or selecting experiments."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from opd_pilot.evidence import publish, separate_output
from opd_pilot.update_probe import digest, file_digest, require

FIELDS = (
    "hypothesis",
    "scope",
    "holdout_tasks",
    "exploration_tasks",
    "prefix_rule",
    "candidate_rule",
    "seeds",
    "primary_metrics",
    "budget",
    "exclusions",
    "statistics",
    "interpretation",
    "evidence_references",
    "approval_reference",
)


def task_ids(path):
    rows = [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [row["task_id"] for row in rows]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Task list is empty or contains duplicate IDs")
    return ids


def freeze(protocol, output):
    require(protocol, FIELDS)
    if protocol.get("confirmation_results_seen") is not False:
        raise ValueError(
            "Confirm that this protocol is locked before reviewing confirmation results"
        )
    holdout = task_ids(protocol["holdout_tasks"])
    explored = task_ids(protocol["exploration_tasks"])
    if set(holdout) & set(explored):
        raise ValueError("Confirmation tasks overlap exploration tasks")
    require(protocol["budget"], ("tasks", "repeats", "max_env_steps", "max_new_tokens"))
    if any(type(value) is not int or value <= 0 for value in protocol["budget"].values()):
        raise ValueError("Budget entries must be positive integers")
    if protocol["budget"]["tasks"] != len(holdout):
        raise ValueError("Freeze the exact chosen task list, matching the task budget")
    require(protocol["seeds"], ("environment", "candidate", "continuation"))
    if any(type(value) is not int for value in protocol["seeds"].values()):
        raise ValueError("Seeds must be explicit integers")
    for name in FIELDS:
        if not protocol[name]:
            raise ValueError(f"Unfilled confirmation field: {name}")
    output = separate_output(output, protocol["holdout_tasks"], protocol["exploration_tasks"])
    locked = dict(protocol)
    locked.update(
        task_ids=holdout,
        task_files={
            name: file_digest(protocol[name]) for name in ("holdout_tasks", "exploration_tasks")
        },
        locked_at=datetime.now(timezone.utc).isoformat(),
        label="confirmation_protocol",
        execution_authorized=False,
        declaration_scope="User declaration; cannot prove results were never viewed",
    )
    publish(output, {"protocol.json": locked, "LOCK.json": dict(digest=digest(locked))})
    return locked


def verify(path):
    path = Path(path)
    protocol = json.loads((path / "protocol.json").read_text(encoding="utf-8"))
    lock = json.loads((path / "LOCK.json").read_text(encoding="utf-8"))
    if digest(protocol) != lock["digest"]:
        raise ValueError("Locked protocol changed")
    for name, expected in protocol["task_files"].items():
        if file_digest(protocol[name]) != expected:
            raise ValueError(f"Task list changed: {name}")
    return dict(status="protocol_integrity_passed", execution_authorized=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("freeze", "verify"))
    parser.add_argument("--spec", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    result = (
        freeze(spec["protocol"], spec["output"])
        if args.operation == "freeze"
        else verify(spec["directory"])
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
