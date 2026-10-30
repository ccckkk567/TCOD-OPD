# Approved recording integration — implemented locally

The user approved observational instrumentation on 2026-10-30, first for ALFWorld
and the common veRL trainer, then explicitly for WebShop and ScienceWorld as well.
The implementation modifies those three OPD workflow files and the actor-update
boundary in `trinity/trainer/verl_trainer.py`. The shared adapter is
`opd_pilot/integration.py`. This is not approval for other TCOD behavior changes.

## Recorded evidence and preserved behavior

- Snapshot actual messages and sampling kwargs before each existing student call.
  Record returned tokens, prompt length, response and student logprobs; capture the
  parsed action and original observation/info/reward/done results. Partial turns
  remain identifiable when an environment step or teacher call raises.
- Record scores from the existing teacher call and its actual temperature. Keep
  TCOD's `prompt_length - 1` slice unchanged. No extra generation/scoring calls,
  hidden environment queries, teacher actions or filtering are introduced.
- ALFWorld retains its original done-to-final-reward behavior. WebShop retains its
  action validation and synthesized invalid-action transition, which is labeled
  as having no environment step. Each repeated WebShop rollout gets a separate
  episode. ScienceWorld retains best-score/100 reward and records its existing
  best score separately from the last available info score. Success stays unknown;
  recording does not reinterpret done or reward as a success label.
- Attach a readable `opd_turn_uid` in Experience.info. The runner assigns parts of
  the original EID after workflow return, so transient workflow EIDs cannot serve
  as stable record IDs. Immediately before the existing `_update_actor(batch)`,
  map the final batch's unique_ids back to these records. Preserve row order,
  tensors, original IDs, loss, optimizer and update count.
- Capture the batch actually handed to the actor, its response mask, advantages,
  masked-token count and runner-supplied sampling model versions. This observes
  the actor-call boundary, not individual optimizer microsteps inside veRL.
  Emit `update_attempt` first, `update` only after normal return, or `update_failure`
  if the call raises. A failure can follow a partial update; no rollback is claimed.

## Enabling and provenance

Recording is off when `OPD_RECORD_ROOT` is absent or empty. Enabling it requires
all three environment variables in both trainer and workflow worker processes:

- `OPD_RECORD_ROOT`: a fresh run's records directory under
  `/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/outputs/tcod-opd/`.
- `OPD_STUDENT_REVISION`: declared student base-weight revision from asset metadata.
- `OPD_TEACHER_REVISION`: declared teacher base-weight revision from asset metadata.

These are metadata labels, not a claim that the adapter verified loaded weights.
The trainer records actual runner-assigned sampling version counters separately.
A workflow-local sampling counter may be null because the runner adds it later.
`trainer_step_before` and `trainer_step_after_if_completed` are trainer counters,
not checkpoint identities or proof of a particular live weight version.

Use the same values in the approved launch environment and `explorer.env_vars`.
TCOD's existing Config.get_envs does not automatically forward these custom names.
Set them before starting a dedicated Ray cluster on every participating node;
setting them in a shell after connecting to an existing cluster is insufficient.
The eventual GPU smoke must verify both trainer and explorer shards. Current
generated configs and queued jobs are not silently enabled or changed here.

The readable host/process/session names are exclusive and survive PID reuse by
advancing to the next unused name. No hashes are added to evidence filenames.
Each run should use a separate root; audit/merge only after all writers stop.

## Errors, overhead and remaining validation

The adapter propagates writer exceptions. At the workflow boundary this fails the
task, but TCOD's original WorkflowRunner catches ordinary exceptions and returns
Status(False). Therefore **this integration does not guarantee immediate exit of
the entire Slurm job**. The earlier proposal's job-wide wording was too strong.
The runner/scheduler exception policy has not been changed. Making every failed
record stop the whole distributed run would require a separately reviewed change.

COMPLETE means the writer drained, not that an episode or actor update succeeded.
Hard termination may leave an incomplete shard; audit refuses such a shard.
An attempt without a terminal event is reported as unresolved, never successful.
Array snapshots, JSON serialization, per-episode flush/fsync and bounded writer
backpressure add overhead. In asynchronous training this can affect timing,
arrival order and staleness; unchanged algorithms do not imply identical runs.

Tests execute production workflow methods with deterministic model/environment
doubles and the actual actor-call with-block. They compare calls, inputs, rewards,
teacher scores and outputs with recording disabled/enabled; cover invalid actions,
partial failures, post-run EIDs, reordered batches and visible write failures.
These tests do not exercise Ray/CUDA, actual benchmark packages or veRL internals.
Real tensor/info serialization, worker propagation, scoring alignment, overhead
and checkpoint recovery still require compute-node validation.
