#!/bin/bash
#SBATCH --job-name=opd-alf-context-check
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=01:00:00
#SBATCH --signal=B:USR1@600
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/train/%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
config=${1:?config path required}
revision=${2:?expected code revision required}
test "$(git rev-parse HEAD)" = "$revision"
test -z "$(git status --porcelain --untracked-files=no)"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s -n 9 || { echo 'Shared environment is being modified'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
source "$CREDENTIALS_FILE"
# Project is supplied explicitly by the monitor. Some SDK settings versions
# interpret SWANLAB_PROJECT as structured JSON rather than a plain project name.
unset SWANLAB_PROJECT
# H200 is CUDA. Preserve Slurm's CUDA assignment; inherited ROCm selectors make
# veRL reject worker initialization when CUDA_VISIBLE_DEVICES is also present.
unset ROCR_VISIBLE_DEVICES HIP_VISIBLE_DEVICES
export PYTHONPATH="$PWD:$ORION_ROOT/code/webshop"
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="/tmp/opd-work-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
python -m orion.scripts.alfworld_context_smoke --config "$config" \
  --output "$REPORT_ROOT/alfworld-context-h2-acceptance"
