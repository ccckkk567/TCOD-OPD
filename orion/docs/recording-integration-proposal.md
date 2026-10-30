# Proposed ALFWorld training record integration — awaiting user approval

The current source performs student chat generation, environment steps and teacher
scoring in `trinity/common/workflows/envs/TCOD/alfworld/OPD_workflow.py`. It does not
persist the independent recorder schema. In
`trinity/trainer/verl_trainer.py`, `train_step` computes advantages then calls
`_update_actor(batch)`. The converted batch already carries `unique_ids` and model
versions, allowing records to follow the actual batch rather than assumed order.

Proposed scope, not yet implemented:

1. Add opt-in recording controlled by an explicit record-root setting. Capture the
   exact messages before the existing chat call, the returned tokens/logprobs,
   parsed action and the raw environment return. After the existing teacher call,
   record those actual teacher scores. Keep original prompts, action parsing,
   environment wrappers, done handling, final reward and teacher slicing unchanged.
   No extra teacher generation or scoring calls are introduced.
2. Associate records with existing experience unique IDs and recorded sampling
   versions. Unknown success/outcome fields stay unknown; do not request new hidden
   environment information or reinterpret the author's reward.
3. At the real trainer update boundary, capture the actual consumed unique IDs,
   response masks, computed advantages, valid-token count and version/update IDs.
   Preserve batch order, optimizer, learning rate, scheduler, loss and update count.
   Record whether the existing update completed or raised, without reporting an
   attempted update as successful. The recorder schema will be extended explicitly
   for attempted/failed updates rather than overloading successful update records.
4. Use separate worker/session evidence files under storage_ext outputs/tcod-opd.
   Enable/disable recording through the approved run setup. Copies, bounded writer
   backpressure and disk IO can add overhead. With recording enabled, a writer
   failure will stop the affected run rather than silently losing critical records.

Files proposed for source instrumentation: the ALFWorld OPD workflow and
`trinity/trainer/verl_trainer.py`. Independent adapter/record-schema additions would
live under opd_pilot. Tests will verify the same existing model/env/update calls
and inputs, correct IDs after batch reordering, and failure reporting. Real
performance/serialization correctness still needs a short GPU smoke.

This approval would not authorize reward fixes, teacher actions in training,
new filtering, 50-step/scheduler changes, new GPU layouts or changes to other
benchmark workflows. Those remain separate decisions. No source instrumentation
has been applied by writing this proposal.
