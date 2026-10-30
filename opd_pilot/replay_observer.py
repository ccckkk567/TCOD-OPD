"""Opt-in E5 worker preflight and actual optimizer-step evidence."""

import json
import os
from pathlib import Path


def enabled():
    return bool(os.environ.get("OPD_E5_REPLAY_ROOT"))


def write(name, value):
    root = Path(os.environ["OPD_E5_REPLAY_ROOT"])
    root.mkdir(parents=True, exist_ok=True)
    with (root / (name + ".json")).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)


def single_step_errors(rows, mini_batch_size, epochs, sequence_parallel):
    errors = []
    if rows != mini_batch_size or epochs != 1:
        errors.append("Native rank-local batch/epochs do not yield exactly one optimizer step")
    if sequence_parallel != 1:
        errors.append("Replay sample-order audit currently requires the original SP=1 topology")
    return errors


def collective_require(errors):
    import torch
    import torch.distributed as dist

    flag = torch.tensor(int(bool(errors)), device=torch.cuda.current_device())
    dist.all_reduce(flag, op=dist.ReduceOp.MAX)
    if flag.item():
        raise ValueError("E5 preflight failed on at least one rank: " + str(errors))


class ActorTrace:
    def __init__(self, actor, data):
        import torch.distributed as dist

        self.rank = dist.get_rank()
        self.actor = actor
        self.ids = data.non_tensor_batch["unique_ids"].tolist()
        errors = single_step_errors(
            len(data),
            actor.config.ppo_mini_batch_size,
            actor.config.ppo_epochs,
            actor.ulysses_sequence_parallel_size,
        )
        if not hasattr(actor.actor_optimizer, "register_step_post_hook"):
            errors.append("Optimizer cannot provide actual step evidence")
        if getattr(actor, "scaler", None) is not None:
            errors.append(
                "Replay cannot certify restored GradScaler state; keep original BF16 configuration"
            )
        collective_require(errors)
        self.attempts, self.executed = 0, 0
        self.micro_batches = []
        self.handle = actor.actor_optimizer.register_step_post_hook(self._executed)

    def _executed(self, optimizer, args, kwargs):
        self.executed += 1

    def micro_order(self, indices):
        self.micro_batches = [[self.ids[int(i)] for i in group] for group in indices]

    def finish(self, scheduler_before, scheduler_after, error=None):
        self.handle.remove()
        write(
            f"rank-{self.rank}-update",
            dict(
                rank=self.rank,
                sample_ids=self.ids,
                micro_batches=self.micro_batches,
                optimizer_step_attempts=self.attempts,
                optimizer_steps_executed=self.executed,
                scheduler_before=scheduler_before,
                scheduler_after=scheduler_after,
                scheduler_steps=scheduler_after - scheduler_before,
                error=error,
                status="returned" if error is None else "failed",
            ),
        )


def restored(worker, path):
    """Verify extra state presence and loaded scheduler/RNG immediately after native restore."""
    import torch
    import torch.distributed as dist

    from opd_pilot.native_codec import encode
    from opd_pilot.update_probe import digest

    rank, world = dist.get_rank(), dist.get_world_size()
    extra = torch.load(
        Path(path) / f"extra_state_world_size_{world}_rank_{rank}.pt", weights_only=False
    )
    scheduler = worker.actor_lr_scheduler.state_dict()
    rng = worker.checkpoint_manager.get_rng_state()
    errors = []
    if extra.get("rng") is None or extra.get("lr_scheduler") is None:
        errors.append("Checkpoint lacks scheduler or RNG state")
    else:
        if digest(encode(extra["lr_scheduler"])) != digest(encode(scheduler)):
            errors.append("Scheduler restoration mismatch")
        if digest(encode(extra["rng"])) != digest(encode(rng)):
            errors.append("RNG restoration mismatch")
    if not worker.actor_optimizer.state:
        errors.append("Restored optimizer has no state")
    write(
        f"rank-{rank}-restore",
        dict(
            rank=rank,
            errors=errors,
            scheduler=encode(scheduler),
            rng_digest=digest(encode(rng)),
            optimizer_state_entries=len(worker.actor_optimizer.state),
        ),
    )
    collective_require(errors)


def prepared_batch(batch, step):
    if not enabled():
        return
    from opd_pilot.native_codec import encode

    write(
        "prepared-batch",
        dict(
            outer_train_step=step,
            tensors=encode(dict(batch.batch.items())),
            non_tensor=encode(batch.non_tensor_batch),
            metadata=encode(batch.meta_info),
        ),
    )
