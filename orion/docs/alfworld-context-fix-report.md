# ALFWorld per-step context correction

Implementation base: `2bf9756` on `codex/orion-opd`. Upstream audit reference: `465eef4406ad0cff675b36bd46f37f28b1736ff9`.
Source: https://github.com/kokolerk/TCOD/issues/9 (owner replies June 28 and July 4).
The paper's full-history mathematical state does not establish its actual prompt rendering. Owner replies specify independent prompts and fixed history during training and evaluation. This is a reconstruction from those replies and the approved plan, not an official author patch.

## Changes and actual entry points

- `trinity/common/workflows/envs/TCOD/alfworld/context.py`: pure complete-template builder, fixed last two observation/action pairs, empty history on first turn.
- `OPD_workflow.py`: existing `OPD_alfworld_workflow` registry class; reset validates mode on every task, defaults to legacy. Only `per_step_h2` replaces messages each turn. Teacher receives unchanged sampled IDs and slices from P-1; reward, parser, EID, filtering, advantage, optimizer and sampling order unchanged.
- `opd_pilot/environment.py`, `evaluate.py`: student, teacher and checkpoint diagnostic evaluation share the builder when spec explicitly selects per_step_h2. The Transformers backend remains an independent evaluator, not a claim of vLLM sampling equivalence. Generic `alfworld_workflow` is not used.
- `continuation.py`: accepts the explicit mode; existing exact prompt/state checks reject mismatched replay. Existing scores/probes consume recorded tokens/messages and must be regenerated from the new run. Legacy caches are not relabelled.
- `alfworld_baseline_specs.py`: `--context-mode per_step_h2` prepares corrected student/teacher specifications with the same 32-task selection, seed, repeats and generation settings. Existing legacy specs stay legacy; do not reuse their outputs as corrected baselines. No additional baseline jobs are submitted by this patch.
- `context_audit.py`, `train.py`, `integration.py`: config protocol consistency, tokenizer vocab/added/special-ID compatibility, source/template fingerprints, resolved configuration and per-turn truncation/mask evidence. No hashes in filenames. Training-segment acceptance fails after cleanup if dummy/no-valid-token evidence exists; dependent stages cannot expand a failed segment. This does not change the production overflow policy mid-episode.
- `alfworld_context_smoke.py`: real native workflow + vLLM shared inference; two allocated H200s, original student/teacher memory budgets on the inference card; no trainer instantiated. Two existing train tasks, original 30-turn cap. Guard checks before parsing/action, same teacher IDs and P-1 scoring alignment, evaluator/student prefix token equality, budgets, mask and response lengths. Failed task and partial evidence remain visible. No success-rate claim.

## New chain

`orion/scripts/submit_context_recovery_chain.sh` submits four dependent jobs:
1. Native no-update acceptance (2 H200, 1h).
2. Fresh 2 updates (2 H200, existing 4h budget).
3. Restore and update to step 4 (2 H200, existing 4h budget).
4. Previously authorized replacement 50-update pilot, fresh initial student, 6h; existing recovery gate before training.

All dependencies use afterok and kill-on-invalid-dep. A failure blocks later work. Old cancelled job 182988 and legacy checkpoint are not reused.
The attachment's eight-GPU statement is superseded by the approved two-H200 layout. The already authorized replacement pilot is queued, not a new study or ablation.

Names: `alfworld-s17-two-gpu-smoke-context-h2` and `alfworld-s17-two-gpu-pilot-context-h2`. Each has its own Ray namespace, queue buffer name, experience path, worker temp directory, output, checkpoint and records. The 2+2 segments intentionally share their new protocol run, while the pilot starts fresh. No old experience buffers or data/models are copied/deleted.
The shared checkout is frozen using a detached Git worktree (no new clone). Existing shared environment is read-locked and never reinstalled.

Research settings retained: Qwen3-1.7B / Qwen3-8B; student prompt 10240 and response 512; teacher input 16384, total 20480; 30 environment turns; original tasks/seed/temperature/top-p/top-k/thinking/template; asynchronous sampling and staleness control; effective train batch 64; original loss, old-logprob computation and optimizer; LR horizon 100; pilot save steps 10/25/50. Context mode and run identities are explicit new differences.

## Verification

Local: 25 targeted tests passed after final review, including four-turn windows, 30-turn simulation, prepatch legacy equality, reset/mode rejection, exact teacher IDs and position-distinct scoring, evaluator message equality, real CPU Experience masks/padding, dummy guard, observer linkage and existing evaluator/replay tests. Final verification is recorded in the delivery message.
`orion/docs/alfworld-context-input-evidence.json` contains four actual simulated prompts and token alignment evidence. CPU tests verify shapes/positions, not GPU equivalence.
GPU native smoke, actual token mappings on downloaded models, actual prompt token equality, new 2+2 recovery and 50-update experiment: NOT EXECUTED LOCALLY. HPC is only operated by the user. No training results are claimed.

Remote evidence:
- reports/tcod-opd/alfworld-context-h2-acceptance/{evidence,context-manifest,resolved-config}.json (resolved-config uses .yaml).
- outputs/tcod-opd/<new-run>/<phase>/{context-manifest,context-audit,COMPLETE}.json and records/.
- logs/tcod-opd/train/<job>.log; native smoke Ray logs in logs/tcod-opd/context/<job>/.

## Limits and rollback

Two history entries do not imply fixed token length. Independent observation/action lists may still overflow; do not silently enlarge budgets, omit actions or disable checks. Dummy masks are zero in original Experience; earlier claims of guaranteed dummy-token learning were incorrect. Dummy can still affect environment calls and unmasked workflow KL statistics; this patch preserves that production behavior and audits it.
Done-to-reward=1 interpretation remains upstream; benchmark success is not certified by this context fix. Workflow temperature/max steps reset behavior is unchanged and effective settings are recorded. Teacher/student tokenizer compatibility must pass on HPC. Legacy rollback means selecting `legacy_accumulated` in a separately named run; never resume across context protocols.
