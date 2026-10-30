# Orion Vanilla OPD

Current stage: source environment preparation. User returned completed CPU probe 180494 and GPU probe 180495 on 2026-10-30. H200 and driver 580.126.20 were observed; Python 3.11.9, CUDA 12.6.2 and GCC 10.5.0 modules are available. No training environment is validated and no experiment has run. Network probes showed intermittent timeouts; successful homepage access does not establish model-download or SwanLab authentication success.

The user controls the remote terminal. Local changes travel through the user's GitHub repo; never clone the upstream repo as the HPC experiment checkout.

## Layout

See `site.env` for exact paths and `../AGENTS.md` for standing rules. Models and benchmarks are shared across projects, not nested under TCOD. Names are readable; exact revisions go in metadata. Credentials stay in the existing remote file and never enter Git.

## First wave

Create `logs/tcod-opd` first, then submit `slurm/bootstrap.sbatch` (or paste its body into `sbatch` using a quoted heredoc). This CPU job clones `https://github.com/ccckkk567/TCOD-OPD.git`, branch `codex/orion-opd`, into `code/TCOD-OPD` and submits the two probes. It refuses to overwrite an existing checkout.

For an already cloned checkout, run `bash orion/scripts/submit_probe.sh` on the login node. It submits two short capability checks, on ioqueue and testqueue. Copy back `logs/tcod-opd/probe/cpu-<job>.log` and `gpu-<job>.log`. Checks report available container files, modules, device/driver, network access and directory visibility without installing anything or printing keys. Individual unavailable capabilities are reported, not treated as a training failure. Do not run both submission methods: that would duplicate probes.

## Next stages (not implemented/validated yet)

1. User selected README-based source installation, not a generic Trinity container. `install_env.sbatch` prepares `envs/tcod-opd` on testqueue and runs a basic attention forward/backward check. Full model and OPD checks remain subsequent gates.
2. Download/reuse Qwen3-1.7B, Qwen3-4B, Qwen3-8B and Qwen3-30B-A3B-Instruct-2507 on ioqueue, at most two concurrent downloads.
3. Prepare ALFWorld, WebShop and ScienceWorld independently. Close the ALFWorld loop first.
4. Patch failure propagation, behavior records and SwanLab; validate token scoring and small update/recovery.
5. Baseline evaluation, qualified fixed teacher, then two student training chains and separate diagnostics.

Do not submit placeholder commands from the plan. Do not mark planned tests or experiments as completed. Container recipes in the upstream repo are build instructions, not evidence of an already-published compatible image.

## Source environment installation

On the login node, after earlier probes have finished:

```bash
cd /scratch/prjsnnn0000269/Students/shuming/cjk/code/TCOD-OPD
git pull --ff-only origin codex/orion-opd
mkdir -p /scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/env
sbatch orion/slurm/install_env.sbatch
```

README requests Python 3.10 and editable `.[dev]` plus FlashAttention 2.8.1. Orion has Python 3.11.9, which the project's pyproject supports; use that module and venv under cluster policy. Add the vLLM extra for actual inference. Initial constraints select vLLM 0.10.2 (the source's noted stable choice), veRL 0.7.0 and matching Torch 2.8 CUDA 12.8 wheels. These are provisional compatibility choices, not a recovered paper environment lock.

FlashAttention 2.8.1 has no official Python 3.11/Torch 2.8 wheel in its release assets, so the job builds from source using the available CUDA 12.6 compiler. That compiler/runtime minor difference is explicit and requires GPU validation; if compilation or kernels fail, inspect the log and align the toolkit, rather than silently upgrading packages. No driver is installed or changed. Logs and resolved dependencies are saved under `logs/tcod-opd/env/<job-id>/`. No credential is required for this installation.

Do not pull changes into this editable checkout while installation or training uses it. A successful install writes `ENV_INSTALLED.json`; this is only a basic environment check, not a model-training READY marker. Incomplete installations may be retried after inspecting the failure; published environments are not modified by the installer.

## Local preparation while the install job is queued

Launcher exceptions now propagate to the CLI. Completed Ray refs are retrieved to surface worker errors; a trainer that fails to finish after the explorer exits is a timeout, not success. Trainer-first completion retains the existing short explorer grace period. Shutdown failures also propagate. Eight dependency-free tests exercise production function bodies with fake Ray, including process exit status; they do not replace a real distributed run or checkpoint-completeness check.

The inherited pre-commit workflow previously checked all upstream files, including existing style/type errors. It now checks cumulative changes since the pinned upstream baseline, then runs the small launcher tests and shell syntax checks. This does not certify untouched upstream files. Sphinx built successfully in the initial run but publishing failed with GitHub bot permission denied (403). This experiment branch keeps a manual document build/artifact workflow without automatic website publishing. The untouched `main` branch still carries its original workflow until these changes are merged.

Do not pull these queued local changes on HPC during installation. Synchronize after the current job finishes and before starting the next stage.
