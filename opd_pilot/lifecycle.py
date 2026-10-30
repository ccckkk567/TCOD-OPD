"""Run completion evidence and opt-in cooperative stop requests."""

import json
import os
from pathlib import Path


class TrainingIncomplete(RuntimeError):
    """Training saved at a boundary but did not reach its requested target."""


def stop_requested():
    """Only read the run-specific marker; signal handlers never save checkpoints."""
    marker = os.environ.get("OPD_STOP_FILE")
    return bool(marker and Path(marker).is_file())


def record_completion(status, completed_step, target, checkpoint_saved, error=None):
    """Atomically publish metadata, without claiming checkpoint restoration worked."""
    destination = os.environ.get("OPD_RUN_STATUS")
    if not destination:
        return
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    payload = {
        "status": status,
        "completed_step": completed_step,
        "target_step": target if target != float("inf") else None,
        "checkpoint_save_returned": checkpoint_saved,
        "recovery_validated": False,
        "exception_type": type(error).__name__ if error else None,
    }
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
