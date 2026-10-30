# Orion Vanilla OPD

Current stage: site discovery. No HPC job has been submitted by Codex, no environment is validated, and no experiment has run.

The user controls the remote terminal. Local changes travel through the user's GitHub repo; never clone the upstream repo as the HPC experiment checkout.

## Layout

See `site.env` for exact paths and `../AGENTS.md` for standing rules. Models and benchmarks are shared across projects, not nested under TCOD. Names are readable; exact revisions go in metadata. Credentials stay in the existing remote file and never enter Git.

## First wave

Create `logs/tcod-opd` first, then submit `slurm/bootstrap.sbatch` (or paste its body into `sbatch` using a quoted heredoc). This CPU job clones `https://github.com/ccckkk567/TCOD-OPD.git`, branch `codex/orion-opd`, into `code/TCOD-OPD` and submits the two probes. It refuses to overwrite an existing checkout.

For an already cloned checkout, run `bash orion/scripts/submit_probe.sh` on the login node. It submits two short capability checks, on ioqueue and testqueue. Copy back `logs/tcod-opd/probe/cpu-<job>.log` and `gpu-<job>.log`. Checks report available container files, modules, device/driver, network access and directory visibility without installing anything or printing keys. Individual unavailable capabilities are reported, not treated as a training failure. Do not run both submission methods: that would duplicate probes.

## Next stages (not implemented/validated yet)

1. Select a compatible existing container or prepare an Apptainer environment on testqueue; freeze dependencies after GPU smoke.
2. Download/reuse Qwen3-1.7B, Qwen3-4B, Qwen3-8B and Qwen3-30B-A3B-Instruct-2507 on ioqueue, at most two concurrent downloads.
3. Prepare ALFWorld, WebShop and ScienceWorld independently. Close the ALFWorld loop first.
4. Patch failure propagation, behavior records and SwanLab; validate token scoring and small update/recovery.
5. Baseline evaluation, qualified fixed teacher, then two student training chains and separate diagnostics.

Do not submit placeholder commands from the plan. Do not mark planned tests or experiments as completed. Container recipes in the upstream repo are build instructions, not evidence of an already-published compatible image.
