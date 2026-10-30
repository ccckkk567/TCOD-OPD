#!/bin/bash
#SBATCH --job-name=opd-config-check
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/config/config-%j.log
set -euo pipefail
set +x
config=${1:?Absolute rendered config path required}
output=${2:?New validation artifact directory required}
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$ENV_ROOT/tcod-opd/bin/activate"
python -m opd_pilot.configs --runtime-config "$config" --output "$output"
