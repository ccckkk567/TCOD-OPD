"""Run an approved single-node configuration in a job-owned local Ray instance."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from opd_pilot.checkpoints import inspect
from opd_pilot.diagnostic_backend import model_manifest
from opd_pilot.evidence import read_records


def completion_evidence(
    output, checkpoint, records_root, target, start, world_size, reference_roots=()
):
    """Require actual successful updates and closed evidence before publishing completion."""
    status = json.loads((output / "training-status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed" or status["completed_step"] != target:
        raise RuntimeError("Training did not reach its requested target")
    records = read_records(records_root, reference_roots=reference_roots)
    steps = sorted(row["trainer_step_after_if_completed"] for row in records["update"])
    if steps != list(range(start + 1, target + 1)) or records.get("update_failure"):
        raise RuntimeError("Successful update records do not match the requested segment")
    report = inspect(Path(checkpoint) / f"global_step_{target}", world_size)
    if report["status"] != "structure_complete":
        raise RuntimeError("Final checkpoint structure is incomplete")
    result = dict(
        status="completed",
        actual_updates=len(steps),
        start_step=start,
        final_step=target,
        checkpoint=report,
        recovery_validated=False,
        evidence_counts={kind: len(rows) for kind, rows in records.items()},
    )
    with (output / "COMPLETE.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    return result


def record_runtime(output, job, student, teacher, config):
    versions = {}
    for package in ("torch", "verl", "ray", "vllm", "transformers", "swanlab", "alfworld"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    metadata = dict(
        job_id=job,
        code_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        package_versions=versions,
        student_revision=student["revision"],
        teacher_revision=teacher["revision"],
        resume_from_records=str(output.parent / "smoke-first/records")
        if config.continue_from_checkpoint
        else None,
    )
    (output / "run-metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def run(config_path, validate_only=False):
    import ray
    import torch
    import yaml

    from opd_pilot.context_audit import audit_records, manifest, validate_config
    from trinity.cli.launcher import run_stage
    from trinity.common.config import load_config
    from trinity.utils.plugin_loader import load_plugins

    raw_config = yaml.safe_load(Path(config_path).read_text())
    context_run = validate_config(raw_config)
    job = os.environ["SLURM_JOB_ID"]
    load_plugins()
    config = load_config(str(config_path))
    gpu_count = 2 if config.cluster.shared_inference_gpu else 8
    if config.stages or config.cluster.node_num != 1 or config.cluster.gpu_per_node != gpu_count:
        raise ValueError("Use an approved single-node resource layout")
    if torch.cuda.device_count() != gpu_count:
        raise ValueError(f"Exactly {gpu_count} allocated visible GPUs are required")
    if config.cluster.shared_inference_gpu:
        from trinity.common.shared_inference import validate_shared_inference

        validate_shared_inference(config)
        if any("H200" not in torch.cuda.get_device_name(i) for i in range(gpu_count)):
            raise ValueError("The two-GPU experiment requires two H200 devices")
    student = model_manifest(config.model.model_path)
    teacher = model_manifest(config.explorer.auxiliary_models[0].model_path)
    env = config.explorer.env_vars
    for key in ("TMPDIR", "RAY_TMPDIR"):
        if env.get(key):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
    env.update(OPD_STUDENT_REVISION=student["revision"], OPD_TEACHER_REVISION=teacher["revision"])
    required = (
        "OPD_RECORD_ROOT",
        "OPD_STOP_FILE",
        "OPD_RUN_STATUS",
        "OPD_STUDENT_REVISION",
        "OPD_TEACHER_REVISION",
    )
    os.environ.update({key: env[key] for key in required})
    output = Path(env["OPD_RUN_STATUS"]).parent
    output.mkdir(parents=True, exist_ok=True)
    if Path(env["OPD_RUN_STATUS"]).exists() or Path(env["OPD_STOP_FILE"]).exists():
        raise ValueError(
            "Run control files already exist; inspect the previous run before retrying"
        )
    if validate_only:
        config.checkpoint_root_dir = str(output / ("validation-checkpoints-" + job))
    checkpoint = Path(config.checkpoint_root_dir) / config.project / config.group / config.name
    if not config.continue_from_checkpoint and checkpoint.exists() and any(checkpoint.iterdir()):
        raise ValueError(
            "Fresh training cannot overwrite or silently rename an existing experiment"
        )
    if (
        not validate_only
        and config.continue_from_checkpoint
        and not (checkpoint / "global_step_2").is_dir()
    ):
        raise ValueError("The approved 2+2 smoke requires its first segment checkpoint")
    logs = Path(os.environ["LOG_ROOT"]) / config.name / job
    logs.mkdir(parents=True, exist_ok=False)
    log_target = logs / "trinity"
    log_target.mkdir(parents=True, exist_ok=True)
    config.log.save_dir = str(log_target)
    runtime_env = {**config.get_envs(), **{key: env[key] for key in required}}
    temporary = Path("/tmp") / ("opd-" + job)
    try:
        ray.init(
            address="local",
            namespace=config.ray_namespace,
            include_dashboard=False,
            num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]),
            num_gpus=gpu_count,
            _temp_dir=str(temporary),
            runtime_env={"env_vars": runtime_env},
        )
        config.check_and_update()
        # Validator resets save_dir to checkpoint/log. Keep both actor config and
        # Ray's already-published TRINITY_LOG_DIR on a stable per-job logs path.
        config.log.save_dir = str(log_target)
        config.save(
            str(output / ("validated-preview.yaml" if validate_only else "resolved-config.yaml"))
        )
        if validate_only:
            return
        (output / "step0-model.json").write_text(json.dumps(student, indent=2), encoding="utf-8")
        record_runtime(output, job, student, teacher, config)
        if context_run:
            manifest(
                yaml.safe_load((output / "resolved-config.yaml").read_text()),
                output / "context-manifest.json",
            )
        run_stage(config)
    finally:
        failing = sys.exc_info()[0] is not None
        try:
            ray.shutdown()
            sessions = list(temporary.glob("session_*/logs"))
            for index, source in enumerate(sessions):
                shutil.copytree(source, logs / f"ray-session-{index + 1}", dirs_exist_ok=False)
        except Exception:
            if not failing:
                raise
            print(
                "Ray cleanup/log copy also failed; preserving the primary training error.",
                file=sys.stderr,
            )
    if context_run:
        audit_records(
            env["OPD_RECORD_ROOT"],
            output / "context-audit.json",
            reference_roots=[output.parent / "smoke-first/records"]
            if config.continue_from_checkpoint
            else (),
        )
    completion_evidence(
        output,
        config.checkpoint_job_dir,
        env["OPD_RECORD_ROOT"],
        config.trainer.total_steps,
        2 if config.continue_from_checkpoint else 0,
        config.cluster.trainer_gpu_num,
        reference_roots=[output.parent / "smoke-first/records"]
        if config.continue_from_checkpoint
        else (),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    run(args.config, args.validate_only)


if __name__ == "__main__":
    main()
