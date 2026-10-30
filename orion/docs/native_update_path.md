# E5 local native-path audit and proposed patch — 2026-10-30

This audit is not validation of Orion's installed runtime. After the offline tools,
the user explicitly approved the isolated native replay implementation below on
2026-10-30. Formal A/B and E6 runs remain separate later decisions.

## Verified local path

1. `trinity/trainer/verl_trainer.py:452` converts Experience using `to_data_proto`,
   post-processes, increments global_steps, optionally balances, and sets temperature.
   `rollout_correction.bypass_mode` selects rollout probabilities; otherwise the
   trainer calls `_compute_old_log_prob` and merges recomputed results. The original
   advantage function then runs, followed by configured rollout correction and actor
   update. Replay must not bypass these operations merely because scores are cached.
2. `trinity/algorithm/advantage_fn/on_policy_distill_advantage.py` computes the original
   teacher-minus-old signal and effective response/teacher masks. Do not rewrite it
   in an offline executor. Invalid score bundles are rejected, never intersected.
3. `trinity/trainer/verl/fsdp_workers.py:214` normalizes mini-batch size by the data
   parallel mesh accounting for sequence parallelism. Global YAML batch size alone
   cannot establish a one-step update.
4. `trinity/trainer/verl/dp_actor.py:95` splits data into mini-batches, loops over PPO
   epochs and mini-batches, then uses dynamic or fixed micro-batches and accumulates
   gradients. `_optimizer_step()` is invoked at line 221 for each mini-batch.
   Its implementation is inherited from the installed veRL base actor; actual
   nonfinite-gradient/skipped-update behavior is not established by this source audit.
5. `trinity/trainer/verl/fsdp_workers.py:861` calls `actor.update_policy` and advances
   the scheduler once at line 894 after that call. An outer train step, worker actor
   call, logical optimizer step and scheduler step are different counters.
6. Existing `update_trace` surrounds the actor call in the driver; it does not prove
   the number of successful optimizer steps inside worker processes. FSDP save/load
   uses the existing checkpoint manager. Archive structure and digests do not prove
   optimizer/scheduler/RNG restoration or asynchronous writer completion.

## Approved native replay implementation

- Add an explicit, isolated replay execution mode only after approval. Construct the
  native wrapper, restore a stopped complete checkpoint, reconstruct Experience via
  a tested lossless adapter, and call the original `train_step` once. No online buffer
  connection and no teacher-generated training trajectories. Preserve all fields,
  order, masks, log-prob modes, precision and resource topology.
- Replace only the explicitly selected teacher score input. Source model, optimizer,
  scheduler/RNG, batch and resolved configuration must match between branches. No
  normalization, historical context changes, filtering or silently changed PPO epochs.
- Instrument native worker boundaries to record epoch/mini/microbatch ordering and
  sample IDs, optimizer attempts/successes, scheduler steps and parameter/gradient
  summaries. Aggregate synchronous rank observations as one logical update, not a
  sum across ranks. Exact skipped-update detection must follow the installed base
  actor implementation; do not equate a function return with a successful update.
- Prefer a naturally one-step native call, established by resolved rank-local batch
  layout. If it cannot be obtained without changing the original configuration,
  report that fact before executing. Do not truncate the optimizer loop. Propose a
  native-update-block comparison or separately approved synchronized single-step
  stopping path. The latter needs an explicit scheduler/checkpoint boundary design.
- Inherit the 100-step scheduler horizon. The no-update reference uses the source
  model without calling any optimizer. Outputs are separate diagnostic artifacts,
  never automatically resumed into or merged into main training.
- HPC acceptance: lossless native conversion, original-versus-replay parity, actual
  restore evidence, same cached scores A/A, actual optimizer count, export/reload and
  source isolation. Numerical differences are measured, not assumed zero.

Implementation: native_replay.py, native_codec.py and replay_observer.py, plus opt-in
hooks in the three native trainer/actor/worker files. Original loss/advantage and
optimizer loops remain intact. The standalone controller restores through the native
loader but omits online explorer weight-sync setup. It owns a private synchronizer
for checkpoint bookkeeping; no main training buffer or explorer is started.

Runtime source checked for the inherited loader and optimizer implementation:
https://github.com/volcengine/verl/blob/v0.7.0/verl/utils/checkpoint/fsdp_checkpoint_manager.py
https://github.com/volcengine/verl/blob/v0.7.0/verl/workers/actor/dp_actor.py
The loader restores extra RNG/scheduler state when present; the replay hook requires
both and compares immediately restored values. Actual optimizer execution is counted
with a post-step hook, so a skipped call is not accepted as an executed step.

The first supported execution mode is engineering-aa, using identical cached native
teacher scores. Native rank-local batch size must equal one mini-batch, PPO epochs=1
and SP=1; collective preflight rejects other layouts before gradient updates. This
does not automatically adjust the pilot batch or permit multi-update blocks. The
original scheduler remains after the complete actor call, retaining horizon 100.

Required real evidence remains: production capture, native conversion/backend parity,
GPU restoration and A/A numerical comparison, final export/reload. CPU tests execute
the actual local actor loop with lightweight model/data adapters and a real optimizer;
they are not distributed GPU validation. No real E1–E4 findings select an intervention.
