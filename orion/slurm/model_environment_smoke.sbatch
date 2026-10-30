#!/bin/bash
#SBATCH --job-name=opd-model-interface
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/model-interface/check-%j.log
set -euo pipefail
: "${SLURM_JOB_ID:?Submit through Slurm}"
benchmark="${1:?Specify alfworld or scienceworld}"
case "$benchmark" in alfworld|scienceworld) ;; *) exit 2 ;; esac
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -s -n 9 || { echo 'Shared environment is being modified; retry after installer finishes.'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$PWD"
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
export TMPDIR="$ORION_ROOT/tmp/$benchmark-model-interface-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
output="$REPORT_ROOT/model-interface/$benchmark-$SLURM_JOB_ID"
printf 'benchmark=%s job=%s host=%s output=%s\n' "$benchmark" "$SLURM_JOB_ID" "$(hostname)" "$output"
# Separate processes guarantee that student GPU allocations end before loading the teacher.
for phase in student teacher; do
  python -m orion.scripts.model_environment_smoke "$benchmark" --phase "$phase" \
    --data-root "$DATA_ROOT" --models-root "$MODEL_ROOT" \
    --alfworld-tasks "$OUTPUT_ROOT/protocols/alfworld/pilot-v1/smoke.jsonl" \
    --output "$output"
done
