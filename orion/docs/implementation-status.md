# Implementation status — 2026-10-30

Local implementation and real experiment completion are separate. No training,
evaluation, export or recovery success is inferred from unit tests. AGENTS.md and
the user's explicit approvals override the reference plan.

## Implemented local path

| Plan area | Implemented entry points | Remaining evidence |
|---|---|---|
| Assets/environment | assets, shared-env preparation, WebShop index and official spaCy resource | Remote package locks, assets, simulator reset/step |
| Split protocols | prepare for ALFWorld; task_protocol for declared WebShop/ScienceWorld training manifests | Runtime tasks and simulator identity |
| P00/P04 completion/stop | Trainer propagation/stop; train driver and Slurm script; checkpoints and export | Distributed failure/stop/save, actual tensors |
| P01/P02 recording | Approved three-workflow and actor-update hooks; records, evidence, merge_records, compact | Actual worker tensors/info, scoring positions, IO/staleness overhead |
| P03 monitoring | Patched SwanLab with local preservation/offline behavior; approved configs | Installed SDK behavior on Orion |
| E1 configuration | pilot_config; six s17/s04 smoke-first, smoke-resume and pilot configs | Actual loader validation and 2+2 smoke |
| E0 evaluation | environment and evaluate; teacher-free; records feed probes | Real runs and serving-backend comparison |
| E2 prefixes/candidates | probes for each fixed/dynamic invocation, pool with provenance, candidates exact-token generation/scoring | Runtime frozen banks and numeric checks |
| E3 continuation | Full-response replay and exposed-state verification, both executors for 2x2 | Simulator reliability and repeated branches |
| E4/reporting | analyze, paired_stats, branch_analysis, selection_analysis, report; CSV/JSON/PNG and cases | Actual outcomes, uncertainty and defensible conclusions |
| Control plane | jobs DAG, separate jobs, IDs, sacct, afterok/afterany | User submissions and remote artifacts |
| Checks | smoke scoring/recovery; production-method doubles and artifact handoff tests | Real scoring, optimizer/scheduler/RNG and asynchronous resume |

The training driver starts its own single-node Ray instance, forwards recorder
variables, routes logs to logs/tcod-opd, and refuses stale control files. COMPLETE
requires target update records and final checkpoint structure. Structure is not
proof of restoration. Failed/interrupted runs retain evidence without fake success.

Approved configs stop at 2, resume to 4, or run continuously to 50. Their LR horizon
is 100; pilot saves are 10/25/50. Original 8-GPU allocation and other training and
sampling parameters are preserved. Continuous 50-step training does not rely on
an untested 25+25 resume strategy.

## Boundaries and conditional extensions

- Separate benchmark venv/RPC modules are superseded by the user's shared-env
  decision. Diagnostic Episode uses original constructors/prompts/parsers.
- Done/reward, overflow/dummy behavior, sample filtering, teacher slicing, loss and
  optimizer are preserved. Plan suggestions to change these are not authorized
  by general instructions to continue. Unknown success is never invented.
- E5 update_probe now provides offline bundle freezing, score alignment, checkpoint
  structure/content auditing, branch planning and descriptive comparisons. E6
  confirmation provides protocol/task-list locking and integrity verification.
  Subsequently approved native Experience codec, isolated replay and per-worker
  optimizer/scheduler/order hooks are implemented for explicit engineering A/A only.
  Real fresh-data capture, GPU restore/update, backend parity and A/A comparison
  remain unvalidated. Formal E5/E6 runs remain unsubmitted and separately authorized.
  See native-replay-usage.md; no mock update or success-returning execution stub exists.
- No automatic resubmission, background cluster monitor or live-checkpoint export
  is enabled. A new attempt follows inspection of failed/ambiguous registry entries.
  Export only after checkpoint writers have stopped.
- Reports retain missing labels and failures. NLL is not action probability.
  Optional adoption frequencies use actual new samples and exact original-parser
  action equality, recording overlaps. Hidden simulator state and causal learning
  mechanisms are not certified by these tools.

## Observed remotely

- Install job 180496 passed imports, attention math comparison and backward checks
  on H200 before subsequent benchmark dependency installations.
- Qwen3-1.7B completed and READY.json was observed.
- Other current asset/job completion is not established here.
- No real OPD update, baseline evaluation or checkpoint recovery is validated.

See diagnostic-runs.md for CLI contracts and training-readiness.md for approved
configs. Full execution-plan completion requires real Orion results; no missing
stage is replaced by fake data or READY.
