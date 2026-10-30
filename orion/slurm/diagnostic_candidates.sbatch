#!/bin/bash
#SBATCH --job-name=opd-candidates
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/diagnostics/candidates-%j.log
set -euo pipefail
set +x
spec=${1:?Provide the path to a fully resolved candidate diagnostic spec}
test -f "$spec"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
source "$ENV_ROOT/tcod-opd/bin/activate"
test -f "$ENV_ROOT/tcod-opd/ENV_INSTALLED.json"
python -m opd_pilot.candidates --spec "$spec" --validate-only
python -m opd_pilot.candidates --spec "$spec"
