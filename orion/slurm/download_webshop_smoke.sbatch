#!/bin/bash
#SBATCH --job-name=opd-webshop-data
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/webshop-data-%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONHOME PYTHONPATH HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s 9
source "$ENV_ROOT/tcod-opd/bin/activate"
source "$CREDENTIALS_FILE"
mkdir -p "$DATA_ROOT/webshop"
exec 8>"$DATA_ROOT/webshop/download.lock"
flock 8
python orion/scripts/download_webshop.py "$DATA_ROOT/webshop" --source hf-smoke
