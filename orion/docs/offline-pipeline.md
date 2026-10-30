# Offline evidence pipeline

These tools are independent consumers of the proposed record schema. They do not
modify TCOD, launch models, execute candidate actions or create training data.
Real records are not yet being emitted by TCOD: that integration still needs the
user's explicit approval. Local synthetic fixtures are correctness tests only.

All commands run on compute nodes after the source writers have stopped. Use
readable experiment subdirectories under `storage_ext/cjk/{outputs,reports}/tcod-opd`.
Never write outputs inside the source evidence or protocol trees. Outputs are
exclusive: choose a new readable version instead of replacing old results.

## Fixed prefix selection

```bash
python -m opd_pilot.probes --spec /absolute/path/probes.json --validate-only
python -m opd_pilot.probes --spec /absolute/path/probes.json
```

Specification fields:

| Field | Meaning |
| --- | --- |
| records_root | Completed worker shards, in the recorder format |
| protocol_root | Published protocol.json, SPLITS_READY.json and four split JSONLs |
| split | Must be analysis_dev for exploratory probe selection |
| student_revision / teacher_revision | Required exact identities as declared in turn records |
| seed | Explicit selection seed |
| max_prefixes / max_per_task | Positive total and per-task caps |
| output | New output directory |

The selector checks split disjointness and actual task membership, sorts source
turn identities before seeded randomization, then samples in task-balanced rounds.
The pool may be smaller than requested; tasks/turns are never duplicated to pad it.
No success, score, disagreement or teacher-outcome filter is applied. Exact stored
messages and the first prompt_length actual token IDs are copied; text is never
reconstructed from observations/actions. Original episode and turn IDs are retained.
Arrays have not yet moved to compact binary storage, so this is for a small pilot.

Output: prefix_bank.json, spec.json, status.json. This is a random main pool only.
It does not create exploratory high-disagreement pools, a cross-student common
pool, a candidate response bank, environment snapshots or validated replay states.
Repeated identical prefixes from distinct turns are not yet deduplicated.
Do not treat a published prefix pool as ready for same-state causal comparisons.

## Data and declared-revision audit

```bash
python -m opd_pilot.audit --spec /absolute/path/audit.json --validate-only
python -m opd_pilot.audit --spec /absolute/path/audit.json
```

Uses the same fields as prefix selection, except seed and prefix caps are not
required. The declared split may be any of the four pilot splits. It checks
publication markers/counts, duplicate/overlapping task IDs, official training-pool
origin, no declared teacher-success filtering, recorded task membership and exact
turn revision labels. Cross-worker linkage is validated by the recorder audit.
It does not read game files or claim that metadata proves actual loaded weights,
student-only execution, token semantics, environment semantics or recovery.
The report explicitly lists those unverified aspects. Output is audit.json plus
status.json; this is not an overall GO/READY gate.

## Paired task statistics

```bash
python -m opd_pilot.paired_stats --spec /absolute/path/paired.json --validate-only
python -m opd_pilot.paired_stats --spec /absolute/path/paired.json
```

Required fields: left_records, right_records, output, metric, seed, replicates,
context. Use metric task_success, score_final, score_best or raw_score; context
should identify environment, split, protocol, left/right checkpoints and purpose.
The caller must establish comparable protocols/model provenance first. Raw scores
from different benchmarks must not be pooled. Revision labels can differ across
checkpoints but must correspond to the intended comparison.

Each side is averaged within task, then paired on task identity. Differences are
right minus left. The percentile bootstrap resamples whole paired tasks, not turns,
tokens or individual repeated rollouts. Original per-task points, unmatched tasks,
known/missing label counts and technical-failure counts are saved. A task with
unknown metrics on one side is unmatched, never assigned zero. With fewer than
two paired tasks no interval is reported. Small pools should be shown directly.

Both all-submitted-known-metric and explicit-technical-failure-excluded analyses
are emitted together. Unknown technical status remains visible and is not silently
excluded. The former still uses known metric labels: it is not a fabricated full
operational success rate when outcomes are missing. The tool does not label a
result significant, causal, exploratory-confirmed or evidence of learning.

Output: paired_statistics.json, spec.json, status.json. A completed status means
the offline computation completed, not that the experiment or scientific claim
passed a gate. Validation mode performs computation but writes nothing. Failed
publication retains an output.in-progress directory for inspection.
