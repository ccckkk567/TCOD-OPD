#!/bin/bash
#SBATCH --job-name=opd-science-smoke
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/scienceworld-smoke-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s 9
source "$ENV_ROOT/tcod-opd/bin/activate"
export TMPDIR="$ORION_ROOT/tmp/scienceworld-smoke-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
java -version
python orion/scripts/benchmark_runtime_smoke.py scienceworld \
  --data-root "$DATA_ROOT/scienceworld" \
  --output "$REPORT_ROOT/benchmarks/scienceworld-smoke-$SLURM_JOB_ID"
