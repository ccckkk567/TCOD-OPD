# Approved training configuration and readiness

The user approved 2 updates + resume 2 updates, and a continuous 50-update pilot,
with an LR schedule horizon of 100, saves at 10/25/50, SwanLab and recording.
Original allocation remains 4 student rollout + 2 teacher + 2 trainer GPUs.
Other sampling, batch, optimizer, advantage and workflow settings remain upstream.

Six configs are committed under
`orion/generated/alfworld-{s17,s04}-{smoke-first,smoke-resume,pilot}/config.yaml`.
Their sibling changes.json files enumerate differences from the upstream example;
rendering specs are in orion/specs. Older upstream-layout configs retain 250 steps
as references; do not use those for this pilot.

First and resumed smoke share checkpoint/buffer identity, with separate record
and status directories. Pilot has a separate run identity. Resume requires the
first segment's global_step_2; emitted update steps are checked after execution.
Failed restoration is not called a successful smoke.

## Runtime entry

On allocated GPUs, `python -m opd_pilot.train --config /absolute/config.yaml`
starts a dedicated Ray instance and invokes TCOD's actual loader, validator and
run_stage. It forwards recording/control variables to workers. `--validate-only`
uses separate validation checkpoint storage and does not train. This path requires
the declared eight visible GPUs; one GPU does not validate this topology.

`orion/slurm/train.sbatch CONFIG FULL_CODE_REVISION` supplies modules, shared venv,
runtime credentials and offline model access. Default request is four hours,
adjustable at submission within 18 hours. The shell handles pre-timeout USR1 by
writing STOP_REQUESTED; saving occurs at the Trainer boundary. Early stop exits
nonzero with partial status. A partial optimizer update is not a certified restart
point even when saving returned.

Logs go to logs/tcod-opd: Slurm per-job logs, Ray session copies, TCOD log symlink
and SwanLab role logs. Checkpoints/evidence stay under storage_ext/cjk. Control
files/output paths are exclusive. Inspect previous results before retries; do not
delete evidence to make a job start.

COMPLETE.json requires expected contiguous successful actor updates, completed
status, linked closed records and all final FSDP rank archives. It does not prove
tensor integrity or optimizer/RNG restoration.

## Remaining runtime gates

1. Finish existing asset/environment jobs without editing their checkout or venv.
2. Validate approved configs on the installed environment and eight-GPU allocation.
3. Run simulator/model evaluation and real scoring comparison.
4. Run smoke-first then smoke-resume; inspect recovery report, LR, teacher identity,
   updates and asynchronous buffer freshness.
5. With usable assets/pairing, submit two continuous pilot chains independently
   within account capacity. Smaller GPU layouts require a separate decision.

These gates are not observed successes. Local implementation submitted no HPC jobs;
the jobs helper never connects remotely.
