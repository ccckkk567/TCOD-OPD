"""Check real recorded scoring and resumed-update evidence on compute nodes."""

import argparse
import json
import math
from pathlib import Path

from opd_pilot.diagnostic_backend import TransformersBackend, model_manifest
from opd_pilot.evidence import publish, read_records, separate_output


def recovery(spec):
    """Require actual consecutive update records and saved-state continuity evidence."""
    import torch

    before = read_records(spec["before_records"])
    after = read_records(spec["after_records"], reference_roots=[spec["before_records"]])
    left, right = before["update"], after["update"]
    start, target = spec["resume_step"], spec["target_step"]
    if not left or not right:
        raise ValueError("Both runs must contain successful actor updates")
    if max(row["trainer_step_after_if_completed"] for row in left) != start:
        raise ValueError("Before-run update boundary differs from resume step")
    steps = sorted(row["trainer_step_after_if_completed"] for row in right)
    if steps != list(range(start + 1, target + 1)):
        raise ValueError("Resumed update sequence skipped or repeated steps")
    if {row["update_uid"] for row in left} & {row["update_uid"] for row in right}:
        raise ValueError("Update record identities were reused")
    teachers = {
        row["scoring_teacher_revision"] for records in (before, after) for row in records["turn"]
    }
    if teachers != {spec["teacher_revision"]}:
        raise ValueError("Teacher identity changed or lacks evidence")
    if any(records.get("update_failure") for records in (before, after)):
        raise ValueError("Recovery smoke contains failed or partially applied updates")
    ranks = []
    for rank in range(spec["world_size"]):
        suffix = f"world_size_{spec['world_size']}_rank_{rank}.pt"
        snapshots = []
        for root in (spec["before_checkpoint"], spec["after_checkpoint"]):
            actor = Path(root) / "actor"
            extra = torch.load(
                actor / ("extra_state_" + suffix), map_location="cpu", weights_only=False
            )
            optim = torch.load(actor / ("optim_" + suffix), map_location="cpu", weights_only=False)
            if not extra.get("rng") or not extra.get("lr_scheduler") or not optim.get("state"):
                raise ValueError("Missing RNG, scheduler or optimizer state")
            counters = {
                str(key): float(value["step"])
                for key, value in optim["state"].items()
                if "step" in value
            }
            snapshots.append(
                dict(
                    epoch=extra["lr_scheduler"]["last_epoch"],
                    counters=counters,
                    learning_rates=[group["lr"] for group in optim["param_groups"]],
                )
            )
            del extra, optim
        a, b = snapshots
        expected_lrs = spec["expected_final_lrs"]
        if len(b["learning_rates"]) != len(expected_lrs) or not all(
            math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-12)
            for actual, expected in zip(b["learning_rates"], expected_lrs)
        ):
            raise ValueError(
                "Restored run's final learning rate differs from the declared schedule"
            )
        if b["epoch"] <= a["epoch"] or set(a["counters"]) != set(b["counters"]):
            raise ValueError("Scheduler/optimizer continuity mismatch")
        if not a["counters"] or any(
            b["counters"][key] < value for key, value in a["counters"].items()
        ):
            raise ValueError("Optimizer counters reset or are absent")
        if not any(b["counters"][key] > value for key, value in a["counters"].items()):
            raise ValueError("No optimizer counter advanced")
        ranks.append(dict(rank=rank, scheduler_before=a["epoch"], scheduler_after=b["epoch"]))
    return dict(
        status="recovery_evidence_consistent",
        ranks=ranks,
        resumed_steps=steps,
        scope="saved_states_and_real_update_records",
        bitwise_recovery_claimed=False,
        unresolved=[
            "live RNG restoration and asynchronous buffer freshness require runtime inspection"
        ],
    )


def scoring(spec):
    manifest = model_manifest(spec["model_path"])
    records = read_records(spec["records_root"])
    turns = {row["turn_uid"]: row for row in records["turn"]}
    backend = TransformersBackend(spec["model_path"])
    comparisons = []
    for uid in spec["turn_uids"]:
        row = turns[uid]
        if (
            row["scoring_teacher_revision"] != manifest["revision"]
            or not row["teacher_scoring_completed"]
        ):
            raise ValueError("Teacher identity/scoring evidence mismatch")
        length = row["prompt_length"]
        actual = backend.score(
            row["tokens"][:length], row["tokens"][length:], row["scoring_temperature"]
        )
        expected = row["teacher_logprobs"]
        if len(actual) != len(expected) or not actual:
            raise ValueError("Scoring token alignment mismatch")
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        passed = all(
            math.isclose(a, b, abs_tol=spec["atol"], rel_tol=spec["rtol"])
            for a, b in zip(actual, expected)
        )
        comparisons.append(
            dict(turn_uid=uid, tokens=len(actual), max_absolute_error=max(errors), passed=passed)
        )
    if not comparisons:
        raise ValueError("Specify actual smoke turn IDs")
    return dict(
        status="passed" if all(row["passed"] for row in comparisons) else "failed",
        comparisons=comparisons,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    operation = spec["operation"]
    if operation not in ("scoring", "recovery"):
        raise ValueError("Choose scoring or recovery")
    required = (
        ("model_path", "records_root", "turn_uids", "atol", "rtol")
        if operation == "scoring"
        else (
            "before_records",
            "after_records",
            "before_checkpoint",
            "after_checkpoint",
            "resume_step",
            "target_step",
            "teacher_revision",
            "world_size",
            "expected_final_lrs",
        )
    )
    for key in required:
        if key not in spec:
            raise ValueError(f"Missing required field: {key}")
    if operation == "scoring" and (not spec["turn_uids"] or spec["atol"] < 0 or spec["rtol"] < 0):
        raise ValueError("Specify real turn IDs and nonnegative score tolerances")
    if operation == "recovery" and not (
        spec["target_step"] > spec["resume_step"] >= 0
        and spec["world_size"] > 0
        and spec["expected_final_lrs"]
    ):
        raise ValueError("Specify valid recovery boundaries, world size and final learning rates")
    sources = [
        value
        for key, value in spec.items()
        if key.endswith(("_records", "_checkpoint", "_path", "_root"))
    ]
    output = separate_output(spec["output"], *sources)
    if args.validate_only:
        print(json.dumps(dict(status="spec_fields_validated", runtime_evidence_validated=False)))
        return
    result = scoring(spec) if operation == "scoring" else recovery(spec)
    publish(
        output,
        {
            "gate_report.json": result,
            "spec.json": spec,
            "status.json": dict(status=result["status"], stage=operation),
        },
    )
    if result["status"] == "failed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
