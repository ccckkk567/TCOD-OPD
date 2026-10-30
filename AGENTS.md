# Orion OPD project rules

- User operates Orion through VS Code. Never SSH to HPC or automate its terminal; provide commands and wait for copied results.
- Make changes locally, commit and push to the user's GitHub repository, then pull on HPC. Upstream kokolerk/TCOD is reference only.
- Keep one working branch `codex/orion-opd` for this experiment series; avoid unnecessary branches.
- Remote root: `/scratch/prjsnnn0000269/Students/shuming/cjk`; Slurm account: `prjsnnn0000269`; user: `linz3`.
- Repositories: `code/`; reusable benchmarks: `data/<benchmark>/`; reusable models: `models/<model-name>/`; environments: `envs/`; containers: `container/`.
- No project-specific nesting for shared benchmarks/models. Check existing assets before downloading. Use readable version names only when versions must coexist, no hashes in directory/file names. Record revisions in metadata.
- Checkpoints and experiment artifacts: `/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/{checkpoints,outputs,reports}/tcod-opd/`.
- Logs: remote root `logs/tcod-opd/<stage>/`; keep per-job logs.
- Credentials are already exported in remote `dep-prep-credentials.env`: `SWANLAB_API_KEY` and `HF_TOKEN`. Source only at runtime with tracing disabled. Never copy, print, commit, or ask user to paste secrets.
- Current user decision: configure from TCOD README using provided Python modules plus venv, since no TCOD-author-validated paper image was identified. Do not substitute a generic Trinity container. Environment installation/compilation belongs on GPU `testqueue`; HF downloads may use CPU `ioqueue`.
- Login node: lightweight file management and Slurm submission/status only. No installs, heavy downloads, imports/tests/training or AI agents.
- GPU queue `testqueue`, CPU queue `ioqueue`. Do not silently switch to other queues. Testqueue jobs must stay within 18h.
- Necessary correctness tests only. Do not change Vanilla OPD algorithm or add teacher actions to student training; diagnostics remain separate.
- Reference plan: `orion/docs/execution-plan.md`. This file and explicit user instructions override its SSH, rsync, directory/hash, CPU-install and venv-first recommendations.
- Never pull into a checkout or modify an environment used by running/queued jobs. Use a readable release tag or separate checkout when necessary; preserve exact commit in metadata.
