# Concrete configuration and checkpoint checks

Two actual path-resolved configurations are committed under
`orion/generated/alfworld-{s17,s04}-upstream-layout/config.yaml`. Both use the
shared Qwen3 student assets, Qwen3-8B teacher and published train_pilot task list.
Each sibling status.json lists every change from the original ALFWorld OPD YAML.
They preserve its 8-GPU topology, 250 steps, save interval, learning rate, sampling
arguments, workflow, advantage and batch sizes. They are not the proposed 50-step
pilot and are not authorized or certified for training submission.

The generator is `python -m opd_pilot.configs --spec <spec.json>`; `--validate-only`
checks rendering without importing TCOD. Existing generated output directories are
exclusive; inspect/reuse committed outputs or choose a new readable directory.
The two specs under orion/specs document how these outputs were built.

Actual validation uses TCOD's `load_config`, plugin loader and
`Config.check_and_update`, not a reimplementation of its schema:

```bash
python -m opd_pilot.configs --runtime-config /absolute/path/config.yaml \
  --output /absolute/path/new-validation-report
```

Run this on a GPU compute node using envs/tcod-opd. The provided
orion/slurm/validate_config.sbatch accepts those two positional paths; create
logs/tcod-opd/config before submitting from the correct checkout. One GPU is
requested for validation/imports, not for running the declared eight-GPU topology.
The validator is permitted to load tokenizers and create metadata directories but
does not start training. It runs offline and may fail until assets/dependencies
exist. The check is implemented but has not been run on Orion.

TCOD's original validator automatically chooses checkpoint/log paths and may
rename existing experiments. To avoid touching real experiment state during this
check, its checkpoint root is temporarily redirected into the validation report.
The resulting validated-preview.yaml is diagnostic only, not a launch config.
Original generated config.yaml keeps the intended storage_ext checkpoint path.
Real training logs still need a reviewed launcher arrangement to route TCOD's
derived log directory into logs/tcod-opd while preserving its original behavior.

## Pending concrete decisions before pilot launch

- The plan proposes 50 updates and intermediate checkpoints; the original example
  uses 250 steps and saves at 250. Changing these and the scheduler requires an
  explicit decision and actual loader validation, not an unnoticed edit.
- Original GPU arithmetic is 4 student rollout GPUs, 2 teacher GPUs, 2 trainer GPUs.
  A smaller Orion layout needs explicit resource choices and a real memory smoke.
- Recording actual messages/tokens/teacher scores and consumed update IDs requires
  instrumentation at real workflow/trainer calls. No such code was changed here.
- These templates retain upstream monitor selection; SwanLab and log routing must
  be made explicit in the eventual approved run configuration.
- No evaluation/recovery success is inferred from this validation or from unit tests.

## Checkpoint structure inspection

`python -m opd_pilot.checkpoints --spec <spec.json>` takes checkpoint, world_size
and output fields. `--validate-only` inspects without publishing a report. Supply
the step directory containing actor/, after writers have stopped. It checks each
model/optim/extra_state_world_size_N_rank_R.pt file using the naming convention in
the current TCOD FSDP checkpoint manager. Missing/truncated archives and unexpected
ranks/world sizes return an incomplete result and nonzero exit code.

The inspector never unpickles tensors. Complete ZIP structure does not prove tensor
contents, optimizer compatibility, scheduler/RNG restoration, buffer cursor recovery
or the next optimizer update. Those remain real recovery-smoke requirements. This
tool does not export a checkpoint or emit a training/model READY marker.
