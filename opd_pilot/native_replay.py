"""Explicit isolated engineering replay through TCOD's native trainer; default dry-run."""

import argparse
import importlib.metadata
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from opd_pilot.checkpoints import inspect
from opd_pilot.evidence import separate_output
from opd_pilot.update_probe import alignment, file_digest, load_bundle, require


def validate(spec):
    require(
        spec,
        (
            "source_config",
            "source_config_sha256",
            "checkpoint_audit",
            "checkpoint",
            "batch",
            "score",
            "output",
            "checkpoint_output",
            "logs",
            "approval_reference",
        ),
    )
    if spec.get("purpose") != "engineering-aa":
        raise ValueError(
            "Only engineering A/A preparation is enabled; formal A/B needs separate approval"
        )
    if file_digest(spec["source_config"]) != spec["source_config_sha256"]:
        raise ValueError("Source resolved configuration changed")
    batch = load_bundle(spec["batch"], "batch")
    scores = load_bundle(spec["score"], "score")
    if not alignment(batch, scores)["aligned"]:
        raise ValueError("Teacher scores are not aligned")
    audit = json.loads(Path(spec["checkpoint_audit"]).read_text(encoding="utf-8"))["checkpoint"]
    if Path(audit["checkpoint"]).resolve() != Path(spec["checkpoint"]).resolve():
        raise ValueError("Checkpoint audit refers to a different source")
    observed = inspect(spec["checkpoint"], audit["world_size"])
    if observed["status"] != "structure_complete":
        raise ValueError("Checkpoint structure incomplete")
    if {item["name"] for item in observed["files"]} != {item["name"] for item in audit["files"]}:
        raise ValueError("Checkpoint audit omits or adds shard files")
    source_unchanged(spec, audit)
    sources = [
        spec[key] for key in ("source_config", "checkpoint", "batch", "score", "checkpoint_audit")
    ]
    outputs = [spec[key] for key in ("output", "checkpoint_output", "logs")]
    for index, path in enumerate(outputs):
        separate_output(path, *sources, *outputs[:index], *outputs[index + 1 :])
    return batch, scores, audit


def source_unchanged(spec, audit):
    for item in audit["files"]:
        if file_digest(Path(spec["checkpoint"]) / "actor" / item["name"]) != item["sha256"]:
            raise ValueError("Source checkpoint changed")


def configure(config, spec, world_size):
    from omegaconf import OmegaConf

    native = OmegaConf.structured(config.trainer.trainer_config)
    if config.trainer.trainer_type != "verl" or native.actor_rollout_ref.actor.strategy not in (
        "fsdp",
        "fsdp2",
    ):
        raise ValueError("Replay currently supports native FSDP veRL only")
    if (
        config.algorithm.algorithm_type != "on_policy_distill"
        or config.algorithm.advantage_fn != "multi_turn_opd"
    ):
        raise ValueError("Only the approved Vanilla multi-turn OPD path is supported")
    if native.critic.enable or config.cluster.gpu_per_node != 8 or config.cluster.node_num != 1:
        raise ValueError("Preserve the approved single-node eight-GPU actor-only layout")
    if native.trainer.nnodes != 1 or native.trainer.n_gpus_per_node != world_size:
        raise ValueError("Checkpoint actor world size differs from resolved topology")
    if native.trainer.total_training_steps != 100:
        raise ValueError("The approved LR horizon must remain 100")
    if not Path(native.actor_rollout_ref.model.path).is_dir():
        raise ValueError("Source model must already exist locally; replay does not download it")
    contents = native.actor_rollout_ref.actor.checkpoint
    for key in ("load_contents", "save_contents"):
        if not {"model", "optimizer", "extra"}.issubset(contents.get(key, [])):
            raise ValueError(
                "Checkpoint configuration must preserve model, optimizer and extra state"
            )
    namespace = "opd-replay-" + os.environ["SLURM_JOB_ID"]
    config.ray_namespace = namespace
    config.synchronizer.ray_namespace = namespace
    native.synchronizer.ray_namespace = namespace
    native.actor_rollout_ref.synchronizer.ray_namespace = namespace
    native.critic.ray_namespace = namespace
    config.log.save_dir = spec["logs"]
    config.checkpoint_job_dir = spec["checkpoint_output"]
    native.trainer.default_local_dir = spec["checkpoint_output"]
    native.trainer.default_hdfs_dir = None
    native.trainer.resume_mode = "resume_path"
    native.trainer.resume_from_path = spec["checkpoint"]
    native.trainer.del_local_ckpt_after_load = False
    config.trainer.trainer_config = OmegaConf.to_container(native, resolve=True)
    return config


class ReplaySession:
    """Ray controller with a private synchronizer for native checkpoint bookkeeping only."""

    def is_alive(self):
        return True

    async def run(self, config, spec, batch, scores):
        from opd_pilot.native_codec import restore
        from trinity.manager.synchronizer import Synchronizer
        from trinity.trainer.verl_trainer import VerlPPOTrainerWrapper

        self.synchronizer = Synchronizer.get_actor(config)
        engine = VerlPPOTrainerWrapper(config)
        # Reuse native preparation operations except explorer weight synchronization.
        engine.actor_rollout_wg.set_algorithm(engine.algorithm_config)
        engine.global_steps = 0
        engine._load_checkpoint()
        start = engine.global_steps
        experiences = restore(batch, scores)
        if any(exp.info.get("model_version") != start for exp in experiences):
            raise ValueError("Batch must be freshly sampled from this exact checkpoint step")
        # A/A uses exactly the cached native scores, not a newly chosen intervention.
        from opd_pilot.native_codec import decode

        for row, exp in zip(batch["rows"], experiences):
            original = decode(row["native_fields"])["teacher_logprobs"]
            if not original.equal(exp.teacher_logprobs):
                raise ValueError("Engineering A/A cannot replace original cached teacher scores")
        metrics = await engine.train_step(experiences)
        evidence = verify_workers(spec["output"], config.cluster.trainer_gpu_num)
        engine.save_checkpoint(block_until_saved=True)
        checkpoint = Path(spec["checkpoint_output"]) / f"global_step_{engine.global_steps}"
        report = inspect(checkpoint, config.cluster.trainer_gpu_num)
        if report["status"] != "structure_complete":
            raise RuntimeError("Replay checkpoint incomplete after native save")
        return dict(
            status="native_replay_completed",
            start_step=start,
            end_step=engine.global_steps,
            logical_optimizer_steps=1,
            worker_evidence=evidence,
            checkpoint=report,
            metrics=metrics,
            formal_experiment=False,
            aa_comparison_validated=False,
            export_reload_validated=False,
        )


def verify_workers(output, world_size):
    rows = []
    for rank in range(world_size):
        row = json.loads((Path(output) / f"rank-{rank}-update.json").read_text())
        if (
            row["status"] != "returned"
            or row["optimizer_step_attempts"] != 1
            or row["optimizer_steps_executed"] != 1
            or row["scheduler_steps"] != 1
        ):
            raise RuntimeError(
                "Native call did not execute exactly one optimizer and scheduler step on every rank"
            )
        rows.append(row)
    return rows


def run(spec, execute=False):
    batch, scores, audit = validate(spec)
    if not execute:
        return dict(
            status="offline_preflight_passed",
            native_update_executed=False,
            pending="HPC resolved configuration, restore and worker checks",
        )
    import ray
    import torch

    from trinity.common.config import load_config

    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.device_count() != 8:
        raise ValueError("Explicit eight-GPU Slurm allocation required")
    if importlib.metadata.version("verl") != "0.7.0":
        raise ValueError("Requires the pinned veRL 0.7.0 runtime")
    config = configure(load_config(spec["source_config"]), spec, audit["world_size"])
    for name in ("output", "checkpoint_output", "logs"):
        Path(spec[name]).mkdir(parents=True, exist_ok=False)
    revision = subprocess.check_output(
        ["git", "-C", str(Path(__file__).resolve().parents[1]), "rev-parse", "HEAD"], text=True
    ).strip()
    (Path(spec["output"]) / "execution-manifest.json").write_text(
        json.dumps(
            dict(
                spec=spec,
                code_revision=revision,
                source_step=Path(spec["checkpoint"]).name,
                package_versions={
                    name: importlib.metadata.version(name) for name in ("torch", "verl", "ray")
                },
                scheduler_horizon=100,
                online_queue_connected=False,
            ),
            indent=2,
        )
    )
    env = dict(config.get_envs(), OPD_E5_REPLAY_ROOT=spec["output"], OPD_RECORD_ROOT="")
    temporary = "/tmp/opd-replay-" + os.environ["SLURM_JOB_ID"]
    try:
        ray.init(
            address="local",
            namespace=config.ray_namespace,
            include_dashboard=False,
            num_gpus=8,
            num_cpus=int(os.environ["SLURM_CPUS_PER_TASK"]),
            _temp_dir=temporary,
            runtime_env={"env_vars": env},
        )
        session = ray.remote(ReplaySession).options(num_cpus=1).remote()
        result = ray.get(session.run.remote(config, spec, batch, scores))
        source_unchanged(spec, audit)
    except BaseException as error:
        (Path(spec["output"]) / "FAILURE.json").write_text(json.dumps(dict(error=repr(error))))
        raise
    finally:
        failing = sys.exc_info()[0] is not None
        if ray.is_initialized():
            try:
                ray.kill(ray.get_actor("synchronizer", namespace=config.ray_namespace))
            except ValueError:
                pass
            ray.shutdown()
        try:
            for index, logs in enumerate(Path(temporary).glob("session_*/logs")):
                shutil.copytree(logs, Path(spec["logs"]) / f"ray-session-{index + 1}")
        except Exception:
            if not failing:
                raise
            print("Ray log copy also failed; preserving original replay error", file=sys.stderr)
    (Path(spec["output"]) / "COMPLETE.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(json.loads(args.spec.read_text()), args.execute), indent=2))


if __name__ == "__main__":
    main()
