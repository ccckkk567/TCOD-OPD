#!/bin/bash
#SBATCH --job-name=opd-env-interface
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/interface-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
benchmark="${1:?Specify alfworld or scienceworld}"
case "$benchmark" in alfworld|scienceworld) ;; *) exit 2 ;; esac
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
# Submission waits for known environment writers; detect later concurrent installs.
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s -n 9 || { echo 'Shared environment is being modified; retry after installer finishes.'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$PWD"
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export TMPDIR="$ORION_ROOT/tmp/$benchmark-interface-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
printf 'benchmark=%s job=%s host=%s\n' "$benchmark" "$SLURM_JOB_ID" "$(hostname)"
python -m orion.scripts.environment_interface_smoke "$benchmark" \
  --data-root "$DATA_ROOT" \
  --alfworld-tasks "$OUTPUT_ROOT/protocols/alfworld/pilot-v1/smoke.jsonl" \
  --output "$REPORT_ROOT/benchmarks/$benchmark-interface-$SLURM_JOB_ID"
