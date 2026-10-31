"""Acceptance checks for the explicitly approved per-step ALFWorld protocol."""

import hashlib
import json
import subprocess
from pathlib import Path

from opd_pilot.diagnostic_backend import token_identity
from opd_pilot.evidence import read_records


def validate_config(config):
    source = config["buffer"]["explorer_input"]
    train = source.get("tasksets") or [source["taskset"]]
    modes = [
        task.get("workflow_args", {}).get("context_mode", "legacy_accumulated") for task in train
    ]
    if "per_step_h2" not in modes:
        return False
    for task in train + source.get("eval_tasksets", []):
        mode = task.get("workflow_args", {}).get("context_mode", "legacy_accumulated")
        default = source.get("default_workflow_type")
        if task in source.get("eval_tasksets", []):
            default = source.get("default_eval_workflow_type") or default
        workflow = task.get("workflow_type") or task.get("default_workflow_type") or default
        if mode != "per_step_h2" or workflow != "OPD_alfworld_workflow":
            raise ValueError("Training/evaluation ALFWorld context protocol mismatch")
    return True


def tokenizers(student, teacher):
    from transformers import AutoTokenizer

    if token_identity(student) != token_identity(teacher):
        raise ValueError("Student/teacher token-ID mappings differ")
    rows = []
    for path in (student, teacher):
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        rows.append(
            dict(
                path=str(path),
                special_tokens=tok.special_tokens_map,
                special_ids=tok.all_special_ids,
                chat_template=tok.chat_template,
                tokenizer_class=type(tok).__name__,
            )
        )
    if (
        rows[0]["special_ids"] != rows[1]["special_ids"]
        or rows[0]["special_tokens"] != rows[1]["special_tokens"]
    ):
        raise ValueError("Student/teacher special-token semantics differ")
    return rows


def manifest(config, output):
    if not validate_config(config):
        raise ValueError("Resolved configuration does not select per_step_h2")
    from trinity.common.workflows.envs.TCOD.alfworld.context import (
        validate_context_mode,
    )

    validate_context_mode("per_step_h2")
    root = Path(__file__).resolve().parents[1]
    files = [
        "trinity/common/workflows/envs/TCOD/alfworld/" + name
        for name in ("context.py", "utils.py", "OPD_workflow.py")
    ]
    report = dict(
        context_mode="per_step_h2",
        history_length=2,
        upstream_reference="465eef4406ad0cff675b36bd46f37f28b1736ff9",
        source="https://github.com/kokolerk/TCOD/issues/9",
        interpretation="Reconstructed protocol; not an official author patch",
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        patch_sha256=hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"])).hexdigest(),
        source_sha256={
            name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files
        },
        config=config,
        tokenizers=tokenizers(
            config["model"]["model_path"], config["explorer"]["auxiliary_models"][0]["model_path"]
        ),
    )
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")


def require_valid_response(response):
    p, n = response.prompt_length, len(response.tokens)
    mask = response.action_mask
    if response.truncate_status == "prompt_truncated" or p <= 0 or n <= p:
        raise ValueError("Context acceptance failed: prompt overflow or empty response")
    if mask is None or len(mask) != n - p or int(mask.sum().item()) != n - p:
        raise ValueError("Context acceptance failed: invalid response mask")
    if response.logprobs is None or len(response.logprobs) != n - p:
        raise ValueError("Context acceptance failed: student logprob alignment")


def audit_records(root, report_path, reference_roots=()):
    rows = read_records(root, reference_roots=reference_roots).get("turn", [])
    errors = []
    for row in rows:
        mask = row.get("action_mask")
        if (
            row.get("context_mode") != "per_step_h2"
            or row.get("truncate_status") == "prompt_truncated"
            or not mask
            or not any(mask)
        ):
            errors.append(row["turn_uid"])
    report = dict(
        status="passed" if rows and not errors else "failed", turns=len(rows), invalid_turns=errors
    )
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["status"] != "passed":
        raise ValueError("Context acceptance failed; see context-audit.json")
    return report
