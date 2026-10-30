#!/bin/bash
#SBATCH --job-name=opd-textworld-meta
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/textworld-metadata-%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock 9
source "$ENV_ROOT/tcod-opd/bin/activate"
python orion/scripts/repair_textworld_metadata.py \
    --backup "$LOG_ROOT/benchmarks/textworld-metadata-$SLURM_JOB_ID"
python -m pip check
python - <<'PY'
import textworld
import alfworld
import scienceworld
print('PACKAGE_IMPORTS_PASSED: textworld, alfworld, scienceworld; simulator reset/step not yet tested.')
PY
