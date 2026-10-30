#!/bin/bash
#SBATCH --job-name=opd-bootstrap
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --chdir=/scratch/prjsnnn0000269/Students/shuming/cjk
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/bootstrap-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Run through sbatch}"
repo=/scratch/prjsnnn0000269/Students/shuming/cjk/code/TCOD-OPD
mkdir -p "$(dirname "$repo")"
if [[ -e "$repo" ]]; then
    echo "Checkout already exists: $repo. Stop to avoid modifying code used by other jobs."
    exit 1
fi
export GIT_TERMINAL_PROMPT=0
git clone --branch codex/orion-opd --single-branch \
    https://github.com/ccckkk567/TCOD-OPD.git "$repo"
git -C "$repo" log -1 --format='Code revision: %H %s'
bash "$repo/orion/scripts/submit_probe.sh"
