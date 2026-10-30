# Independent evidence recorder

`opd_pilot.records` implements the first part of plan P02. It is not connected to
TCOD workflows or the trainer. Any integration that changes how TCOD runs requires
the user's explicit approval of that specific change.

Use an experiment's artifact directory, for example
`/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/outputs/tcod-opd/pilot-01/records/`.
Each worker/session owns a readable, exclusive shard such as
`explorer-1-session-1` or `trainer-1-session-1`. Restart with a new session name;
existing evidence is never overwritten. The caller must give episodes, turns and
updates unique readable IDs across the run.

The API is `Recorder(root, shard, capacity=32)` and
`writer.record(kind, **evidence)`, with `close()` or a context manager. Each instance
belongs to one producer; its background thread drains a bounded queue. Records
are serialized before enqueueing so later mutation of caller data cannot alter
the evidence. Full queues block; write failures surface to the producer or close.
Only pass actual experiment evidence, never credential files or environment dumps.

| Kind | Required evidence |
| --- | --- |
| episode | episode_uid, task_id, outcome |
| turn | turn_uid, episode_uid, sampling_student_revision, scoring_teacher_revision, messages, tokens, prompt_length, response, action, raw_reward, student_logprobs, teacher_logprobs |
| update | update_uid, update_student_revision, consumed_turn_uids, mask, advantage, effective_token_count |

Inputs must be JSON-compatible; callers must explicitly convert tensors to the
appropriate CPU arrays/lists. This module does not infer environment success,
reconstruct prompts, recalculate advantages, select training samples or validate
token-position arithmetic. Unknown outcome fields may remain null.

`events.jsonl` is flushed per record and fsynced on successful close.
`COMPLETE.json` means the writer drained successfully, not that an episode or
training run succeeded. Abrupt process termination can leave an incomplete shard;
it is not silently repaired or admitted to the audit. This is not a distributed
transaction or a guarantee against machine/filesystem failure.

After all writers have closed, run on a compute node:

```bash
python -m opd_pilot.records /path/to/run/records
```

This checks completion counts, duplicate IDs and references from turns to episodes
and updates to consumed turns across shards. It does not merge or modify evidence.

Remaining P02 work: approved integration at actual model/trainer call sites,
compact array storage (NPZ/Parquet), prompt deduplication and offline merge.
Version 1 stores arrays inline in JSONL for a small pilot, so it is not yet the
large-run storage implementation. File and directory names contain no hashes.
