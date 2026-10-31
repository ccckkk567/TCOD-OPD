#!/bin/bash
#SBATCH --job-name=opd-train
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
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
export PYTHONPATH="$PWD:$ORION_ROOT/code/webshop"
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TMPDIR="/tmp/opd-work-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
if [[ -n "${3:-}" ]]; then
    python -m orion.scripts.pilot_recovery_gate \
      --run "${4:-$OUTPUT_ROOT/alfworld-s17-two-gpu-smoke}" --job "$3" \
      --report "$REPORT_ROOT/alfworld-s17-two-gpu-pilot/recovery-gate-$SLURM_JOB_ID.json"
fi
stop_file=$(python - "$config" <<'PY'
import sys, yaml
from pathlib import Path
config = yaml.safe_load(Path(sys.argv[1]).read_text())
print(config['explorer']['env_vars']['OPD_STOP_FILE'])
PY
)
request_stop() {
    wait_interrupted=1
    mkdir -p "$(dirname "$stop_file")"
    touch "$stop_file"
}
trap request_stop USR1
python -m opd_pilot.train --config "$config" &
child=$!
set +e
while true; do
    wait_interrupted=0
    wait "$child"
    status=$?
    # A signal can interrupt wait while the child is still saving.
    if [[ "$wait_interrupted" == 0 ]]; then
        exit "$status"
    fi
done
