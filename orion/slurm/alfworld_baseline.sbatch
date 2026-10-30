#!/bin/bash
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=04:00:00
set -euo pipefail
stage=${1:?prepare or evaluate}
bundle=${2:?absolute bundle directory}
model=${3:-}
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s -n 9 || { echo 'Shared environment is being modified'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$PWD"
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export TMPDIR="$ORION_ROOT/tmp/alfworld-baseline-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
if [[ "$stage" == prepare ]]; then
  python -m orion.scripts.alfworld_baseline_specs \
    --protocol "$OUTPUT_ROOT/protocols/alfworld/pilot-v1" --output "$bundle" --models "$MODEL_ROOT"
elif [[ "$stage" == evaluate && ( "$model" == s17 || "$model" == t08 ) ]]; then
  python -m opd_pilot.evaluate --spec "$bundle/$model.json"
else
  exit 2
fi
