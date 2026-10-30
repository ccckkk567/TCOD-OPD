#!/bin/bash
#SBATCH --job-name=opd-models
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --array=0-3%2
#SBATCH --chdir=/scratch/prjsnnn0000269/Students/shuming/cjk/code/TCOD-OPD
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/models/download-%A_%a.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Run through sbatch}"
source orion/site.env
module purge
module load python/3.11.9
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
source "$ENV_ROOT/tcod-opd/bin/activate"
test -f "$ENV_ROOT/tcod-opd/ENV_INSTALLED.json"
source "$CREDENTIALS_FILE"
export HF_HOME="$ORION_ROOT/downloads/huggingface-cache"
export HF_HUB_ETAG_TIMEOUT=30 HF_HUB_DOWNLOAD_TIMEOUT=60
keys=(s17 s04 t08 t30)
python -m opd_pilot.assets --key "${keys[$SLURM_ARRAY_TASK_ID]}" --models-root "$MODEL_ROOT"
