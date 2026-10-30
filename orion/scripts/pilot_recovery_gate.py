"""Check the real 2+2 artifacts before starting the independent 50-update pilot."""

import argparse
import json
from pathlib import Path

from opd_pilot.diagnostic_backend import model_manifest
from opd_pilot.smoke import recovery


def expected_lr(config):
    optim = config["trainer"]["trainer_config"]["actor_rollout_ref"]["actor"]["optim"]
    if (
        optim["lr_scheduler_type"] != "constant"
        or optim["lr_warmup_steps_ratio"] != 0
        or optim["lr_warmup_steps"] not in (-1, 0)
    ):
        raise ValueError("Recovery gate currently expects constant LR without warmup")
    if optim["lr"] != config["algorithm"]["optimizer"]["lr"]:
        raise ValueError("Resolved optimizer learning rate differs from the approved algorithm")
    return optim["lr"]


def check(run, expected_job, report):
    import yaml

    configs = []
    for phase, start, target in (("smoke-first", 0, 2), ("smoke-resume", 2, 4)):
        directory = run / phase
        complete = json.loads((directory / "COMPLETE.json").read_text())
        if (
            complete["status"],
            complete["start_step"],
            complete["final_step"],
            complete["actual_updates"],
        ) != ("completed", start, target, 2):
            raise ValueError("Smoke completion evidence does not match the 2+2 boundary")
        config = yaml.safe_load((directory / "resolved-config.yaml").read_text())
        if (
            config["cluster"]["trainer_gpu_num"] != 1
            or not config["cluster"]["shared_inference_gpu"]
        ):
            raise ValueError("Recovery evidence is not from the approved two-GPU layout")
        configs.append(config)
    metadata = json.loads((run / "smoke-resume/run-metadata.json").read_text())
    if str(metadata["job_id"]) != str(expected_job):
        raise ValueError("Recovery artifacts belong to a different Slurm job")
    before, after = configs
    teacher = model_manifest(after["explorer"]["auxiliary_models"][0]["model_path"])
    # The approved template uses constant LR with no warmup. Do not derive the
    # expected value from a checkpoint we are trying to verify.
    lr = expected_lr(after)
    if expected_lr(before) != lr:
        raise ValueError("Learning rate configuration changed during recovery")
    spec = dict(
        before_records=str(run / "smoke-first/records"),
        after_records=str(run / "smoke-resume/records"),
        before_checkpoint=str(Path(before["checkpoint_job_dir"]) / "global_step_2"),
        after_checkpoint=str(Path(after["checkpoint_job_dir"]) / "global_step_4"),
        resume_step=2,
        target_step=4,
        world_size=1,
        teacher_revision=teacher["revision"],
        expected_final_lrs=[lr],
    )
    result = recovery(spec)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        json.dumps(dict(recovery_job=expected_job, spec=spec, result=result), indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    check(args.run, args.job, args.report)
