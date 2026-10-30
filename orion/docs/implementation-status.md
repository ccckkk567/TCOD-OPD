# Implementation status

This distinguishes local code, remote evidence and pending integration. It is not
a declaration that the execution plan is complete. No new TCOD source behavior
changes are authorized by the general request to continue local implementation.

## Implemented locally

- Model download/reuse with fixed revisions and size/index checks; GPU/CPU probe
  and source environment setup jobs. Existing remote jobs are user-operated.
- Shared-env benchmark preparation, dependency constraints, serialized installation,
  per-job metadata and GPU recheck. WebShop search index is still pending.
- ALFWorld disjoint pilot split preparation from training games.
- Independent per-worker evidence recorder, failure propagation, offline audit and
  lossless merge. Evidence is not yet emitted by TCOD model/trainer call sites.
- Descriptive outcome analysis with explicit unknown and technical-failure counts.
- Fixed random development-prefix selection from actual recorded messages/tokens.
- Data-membership and declared-turn-revision audit, explicitly limited in scope.
- Task-level paired differences and task-cluster percentile bootstrap, including
  raw per-task points and missing/technical-failure denominators.
- Independent local-asset Transformers candidate generation and full-vocabulary
  temperature-scaled conditional cross-scoring, plus a GPU job entry point. Local
  plumbing tests pass; actual GPU numerics and TCOD-backend equivalence are pending.
- Earlier launcher error propagation and SwanLab offline/local preservation changes
  already committed before the user's new explicit TCOD-change approval rule.

## Observed remotely, not inferred from local tests

- Install job 180496 passed basic imports, attention math comparison and backward
  checks on H200. This was before the subsequent benchmark dependency installations.
- Qwen3-1.7B download job completed and READY.json was observed.
- Other model and benchmark job completion is not established in this document.
- No real OPD update, baseline evaluation or checkpoint recovery has been validated.

## Pending work requiring a concrete integration decision or runtime evidence

- TCOD record emission at actual student/teacher/trainer calls: propose exact files
  and payloads and obtain explicit approval before editing the original execution.
- Preserve TCOD's existing environment/action/reward interpretation. Any proposed
  changes to done/success, teacher scoring or abnormal-sample handling need separate
  evidence and explicit approval. The attempted ALFWorld edit was withdrawn.
- Executable configs: use the real locked TCOD loader and verify on the installed
  environment; do not guess schema fields or silently change upstream algorithms.
- Real token-position/temperature comparison, generation/update/recovery smoke,
  complete checkpoint/export checks and actual independent evaluations.
- Full WebShop dependencies/index/runtime compatibility, real ALFWorld/ScienceWorld
  reset/step checks and record-schema mapping. Shared venvs supersede the plan's RPC
  isolation proposal; a new RPC architecture is not an approved requirement.
- GPU validation of candidate generation/scoring, replay validation, 2x2 continuation
  and update interventions. Standalone candidate code does not execute environment
  actions or demonstrate training-backend equivalence.
- Compact array storage, prefix deduplication, cross-student common pools, plotting,
  case reports, automatic job DAG/reconciliation and final research audit.

No missing stage is replaced by a fake score, fake READY or stub reporting success.
The offline tools can be finished/tested locally, but completing the actual study
still requires remote results and approved TCOD integration.
