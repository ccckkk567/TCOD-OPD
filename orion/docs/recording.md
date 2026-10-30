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

After all producers stop, a separate compute-node job can merge closed shards:

```bash
python -m opd_pilot.merge_records /path/to/run/records /path/to/run/records-merged
```

The destination must be outside the source directory and must not exist. The
merge audits IDs/references first, copies original lines in shard-name order with
bounded copying memory, and audits the result before publishing the destination.
Ordering is deterministic, not chronological. Metadata lists source shard names.
Source evidence remains untouched. A failed merge leaves a readable
`records-merged.in-progress` directory for inspection; it is not silently reused.
The audit retains ID sets in memory, so total memory still grows with record count.

Remaining P02 work: approved integration at actual model/trainer call sites,
compact array storage (NPZ/Parquet) and prompt deduplication.
Version 1 stores arrays inline in JSONL for a small pilot, so it is not yet the
large-run storage implementation. File and directory names contain no hashes.

## Offline outcome analysis

`python -m opd_pilot.analyze --spec /path/to/analysis.json` reads closed shards
and publishes `summary.json`, `report.md` and `status.json` in a separate directory.
`--validate-only` checks the specification and evidence without writing output.
Run on a compute node after all record producers have stopped. No simulator,
model imports or TCOD execution is involved.

The JSON specification requires six nonempty strings: `records_root`, `output`,
`environment`, `split`, `model_revision` and `protocol`. Use one environment,
split, model and protocol per report; those labels are caller-supplied context,
not a substitute for validating provenance at the future integration point.
Existing outputs are never overwritten. A failed write may leave an
`<output>.in-progress` directory for inspection.

The episode `outcome` object may supply `task_success` and `technical_failure`
as true/false/null, `termination_reason` as a string/null, and finite numeric
`score_final`, `score_best`, `raw_score` values (or null). Missing values remain
unknown; neither done nor reward determines success. No final/best score fallback
or benchmark-specific normalization is performed. TCOD integration and mapping
of actual environment fields to this evidence schema remain pending approval.

Reports contain known success/failure/unknown counts, known-label success rate,
label coverage, a separate task-equal known-label success rate, explicit technical
failure counts, termination reasons and independent score denominators. Technical
failures remain in the report. Recorded turn counts are not claimed to be verified
simulator step counts. Scores describe available labels only; unknown outcomes
must not be silently described as failures or omitted from coverage reporting.

This implements descriptive summaries only. Paired statistics, uncertainty
intervals, plots and mechanism claims from plan section 14 remain future work.
