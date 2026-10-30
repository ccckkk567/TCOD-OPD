# Orion OPD project rules

- User operates Orion through VS Code. Never SSH to HPC or automate its terminal; provide commands and wait for copied results.
- Make changes locally, commit and push to the user's GitHub repository, then pull on HPC. Upstream kokolerk/TCOD is reference only.
- Keep one working branch `codex/orion-opd` for this experiment series; avoid unnecessary branches.
- Remote root: `/scratch/prjsnnn0000269/Students/shuming/cjk`; Slurm account: `prjsnnn0000269`; user: `linz3`.
- Repositories: `code/`; reusable benchmarks: `data/<benchmark>/`; reusable models: `models/<model-name>/`; environments: `envs/`; containers: `container/`.
- User approved TCOD's direct-import setup: install benchmark dependencies into the shared `envs/tcod-opd`, not separate benchmark venvs. Serialize shared-environment installs and wait for jobs using it to finish. Preserve reusable data/source assets; only the obsolete benchmark venvs are authorized for removal during migration.
- No project-specific nesting for shared benchmarks/models. Check existing assets before downloading. Use readable version names only when versions must coexist, no hashes in directory/file names. Record revisions in metadata.
- Checkpoints and experiment artifacts: `/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/{checkpoints,outputs,reports}/tcod-opd/`.
- Logs: remote root `logs/tcod-opd/<stage>/`; keep per-job logs.
- Credentials are already exported in remote `dep-prep-credentials.env`: `SWANLAB_API_KEY` and `HF_TOKEN`. Source only at runtime with tracing disabled. Never copy, print, commit, or ask user to paste secrets.
- Current user decision: configure from TCOD README using provided Python modules plus venv, since no TCOD-author-validated paper image was identified. Do not substitute a generic Trinity container. Environment installation/compilation belongs on GPU `testqueue`; HF downloads may use CPU `ioqueue`.
- Login node: lightweight file management and Slurm submission/status only. No installs, heavy downloads, imports/tests/training or AI agents.
- GPU queue `testqueue`, CPU queue `ioqueue`. Do not silently switch to other queues. Testqueue jobs must stay within 18h.
- Necessary correctness tests only. Do not change Vanilla OPD algorithm or add teacher actions to student training; diagnostics remain separate.
- Any change involving how original TCOD works requires advance discussion with the user and explicit approval of that specific change. Explain the original behavior, evidence, proposed change and impact first; only implement after the user explicitly says to change it. Do not infer approval from general instructions to continue the plan, fix issues, prepare experiments or run tests. Suspected bugs, best practices and plan recommendations do not authorize a change. This applies to changes made through wrappers, configuration or scripts as well as edits to upstream source. Read-only investigation and independent preparation that preserves TCOD behavior may proceed without this approval.
- Explicit approval on 2026-10-30: add opt-in observation hooks to ALFWorld, WebShop and ScienceWorld OPD workflows and the shared veRL actor-update boundary. Preserve original prompts, model/environment calls, reward/done interpretation, teacher slicing, filtering and training algorithm. This approval covers recording only, including visible writer failures and IO overhead; other TCOD changes still require separate approval.
- Reference plan: `orion/docs/execution-plan.md`. This file and explicit user instructions override its SSH, rsync, directory/hash, CPU-install and venv-first recommendations.
- Never pull into a checkout or modify an environment used by running/queued jobs. Use a readable release tag or separate checkout when necessary; preserve exact commit in metadata.
