# Independent evaluation, continuation and lifecycle

These modules are implemented locally; real GPU/simulator/Ray validation is pending.
They are independent diagnostics and do not feed the OPD training buffer.

`python -m opd_pilot.evaluate --spec evaluation.json` loads only the selected model.
Required fields: benchmark, model_path, tasks, output, temperature, top_p, top_k
(0), max_new_tokens, max_env_steps, repeats and seed. Optional enable_thinking is
passed to the tokenizer chat template. Tasks is an absolute JSONL path with unique
task_id and task_desc (or ALFWorld game_file) per row. Explicit task descriptions
use the same representation as TCOD: ALFWorld game path, WebShop session number,
or ScienceWorld task_name/var_num configuration. Output is a new artifact directory.

The shared diagnostic Episode imports original TCOD environment constructors,
templates and parsers. Full assistant responses are kept in the history. Tests
compare its prompts and actions with all three production OPD workflows using
deterministic doubles. Generation uses the separate Transformers diagnostic
backend; equality to TCOD's serving backend has not been established. No prompt
truncation is silently applied. Success remains unknown when no explicit label
is available; done and reward are not reinterpreted as success. Exceptions are
retained as technical failures and produce a nonzero process exit after publishing.

`python -m opd_pilot.continuation --spec continuation.json` accepts the same fields,
plus records_root, candidate_bank and executor (student or teacher). The candidate
bank comes from opd_pilot.candidates. One job loads one executor; running the same
student/teacher candidate banks under both executors forms the 2x2 experiment.
Each branch resets the real environment, replays original full responses, compares
historical prompts/transitions and the target exposed observation/info, then injects
the candidate response. Its reasoning as well as its action enters later history.
Mismatch is recorded and never treated as a valid same-state branch. This verifies
exposed state, not invisible simulator internals. Inputs currently require the
chronological fixed-width turn IDs emitted by the approved integration.

Both commands support --validate-only without loading weights or simulators.
The sampling settings, horizon and task split are explicit diagnostic protocols;
they must be matched when comparing runs. Never treat incomplete branches as zero.

## User-operated Slurm helpers

`python -m opd_pilot.jobs validate --spec wave.json` checks an entire dependency DAG.
The wave has repo, revision (full git revision), logs and tasks. Each task has name,
stage, an existing absolute spec path, time (HH:MM:SS), and optional after (job names).
Supported stages are listed in opd_pilot.jobs.STAGES, including train, evaluate,
continuation, candidates, analyze, audit, probes, checkpoints, export, pool, smoke,
webshop_index, task_protocol, paired_stats, branch_analysis, selection_analysis
and report. `submit --spec wave.json --registry new-registry.json` submits separate
GPU testqueue jobs with afterok dependencies and records each returned job ID.
An explicit dependency_mode=afterany is supported for partial reporting. Diagnostic
jobs default to 1 GPU/4 CPUs/32G; train retains 8 GPUs/32 CPUs/256G and takes a YAML
config as its spec path. Memory and time can be specified per task.
The registry must be new; interruption during submission leaves an explicit
uncertain entry for inspection rather than risking duplicate submission.
`reconcile --registry registry.json` performs one sacct query and saves observations.
There is no background process or automatic resubmission. Slurm COMPLETED is not a
claim that the scientific result is valid. These are lightweight login-node commands;
the actual stages run on compute nodes. Keep checkout and specs unchanged while
queued/running jobs use them. The helper never clones or pulls a checkout.

## Approved trainer stop/error handling

Set OPD_STOP_FILE and OPD_RUN_STATUS to paths exclusive to the run in all trainer
processes before starting Ray. Touching the stop file requests a cooperative stop.
The trainer finishes an active update or cancels an outstanding sample wait, then
calls its existing final checkpoint save. A Python signal handler does not perform
collective saves. Partial completion raises TrainingIncomplete after cleanup so
the launcher cannot report it as successful target completion. A short input stream
before a finite target is also partial. Normal target completion is unchanged.

Training and checkpoint errors propagate; the primary training exception is retained
if final save also fails. Status records the last fully returned train step, which
can differ from veRL's counter if the counter advanced before an update raised.
checkpoint_save_returned does not certify restoration or safe reuse after a partial
optimizer failure. Recovery still requires a real smoke. Existing queued jobs are
not enabled or altered by these local changes.

## Artifact contracts and additional stages

All paths in runtime specs must be real absolute paths. Outputs must be new and
outside input evidence/model trees. Heavy work runs on compute nodes. The table
lists JSON fields, not claims that the corresponding assets already exist.

| Module | Input fields beyond output | Published evidence |
|---|---|---|
| task_protocol | benchmark, tasks, seed, counts for train_pilot/analysis_dev/confirm_holdout/smoke | Four disjoint JSONL splits and protocol; input official_split must be train |
| pool | banks (prefix directory list), max_prefixes, seed | Source-balanced common prefix bank with original record roots/IDs |
| export | checkpoint (global_step_N), world_size, run_name | veRL 0.7.0 HF export; tensor/index checks; matching asset.json/READY.json |
| smoke scoring | operation=scoring, model_path, records_root, turn_uids, atol, rtol | Actual recorded teacher scores compared tokenwise to independent backend |
| smoke recovery | operation=recovery, before_records, after_records, before_checkpoint, after_checkpoint, resume_step, target_step, teacher_revision, world_size, expected_final_lrs | Update sequence, saved optimizer counters/scheduler/LR/RNG presence; not bitwise restore proof |
| webshop_index | repo, revision, data_root | Official small-data conversion and Lucene indexes under shared data, linked into source |
| branch_analysis | branch_files, metric | Paired candidate 2x2 summary, prefix CSV, delta scatter, failures/missing counts |
| selection_analysis | candidate_banks, checkpoints, temperature, teacher_revision, metric | Fixed-response NLL, optional measured adoption, continuation by checkpoint |
| report | evaluations, metric; optional branch_files, case_limit, allow_missing | Tables, training/failure chart, contrasting casebook and missing input status |

Every evaluation writes compatible records/ in addition to episodes.json. Probe
specs for these teacher-free records use teacher_revision=null. They do not contain
invented teacher scores. An evaluation still exits nonzero after retaining technical
failures; afterany reporting may summarize these artifacts.

Candidate generation can specify benchmark to attach the original parser's
parsed_action. The generator otherwise retains all complete responses/tokens.
Common-pool candidate specs declare additional_prefix_tokenizer_paths, and their
continuation specs declare additional_records_roots. These checks prevent silently
mixing unrelated record roots or token-ID mappings.

Each selection_analysis checkpoint entry has series, step, revision, scores (score
bank paths), branch_files and optional generated_banks. Fixed candidate responses
must be identical across scoring checkpoints, with matching scoring temperature.
Branch executor revisions are checked against the declared student checkpoint and
fixed teacher. generated_banks must be fresh student samples on those same prefixes
with parsed_action; they measure exact parsed-action frequency, including overlap
between teacher/student candidate action sets. Without samples, adoption is null.
Full-response NLL is reported separately and never called action probability.

Each report evaluation entry has directory, series and step. Model manifests must
match the declared checkpoint step. allow_missing permits missing evaluation paths
for an afterany partial report, records them explicitly and never assigns zero.
Existing malformed data still fails. Branch files must exist; omit failed shards
explicitly and retain the job registry as the failure inventory.

WebShop indexes are a separate GPU preparation job; the benchmark installer adds
the official en_core_web_lg resource to the shared environment. Run both only when
other jobs are not using the affected environment/assets. Reuse checks source
revision and source-file size/mtime and reopens every index; it is not a cryptographic
content audit. Full WebShop remains a separate protocol, never mislabeled as small.

The smoke commands accept --validate-only for required-field/path separation checks;
that mode does not validate GPU execution or load checkpoint payloads. Recovery
loads only this experiment's trusted local torch checkpoints. Export invokes the
installed pinned veRL merger, not a reimplementation of FSDP tensor merging.
