#!/bin/bash
#SBATCH --job-name=opd-replay-aa
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/replay/replay-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
spec="${1:?Specify an explicit engineering replay manifest}"
mode="${2:---validate-only}"
args=()
case "$mode" in --validate-only) ;; --execute) args+=(--execute) ;; *) exit 2 ;; esac
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s -n 9 || { echo 'Shared environment is being modified'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
python -m opd_pilot.native_replay --spec "$spec" "${args[@]}"
