#!/bin/bash
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
stage=${1:?stage required}
spec=${2:?absolute spec required}
revision=${3:?code revision required}
case "$stage" in
  evaluate|continuation|candidates|analyze|audit|probes|checkpoints|export|branch_analysis|pool|smoke|webshop_index|report|task_protocol|paired_stats|selection_analysis) ;;
  *) echo 'Unsupported stage' >&2; exit 2 ;;
esac
test "$(git rev-parse HEAD)" = "$revision"
test -z "$(git status --porcelain --untracked-files=no)"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
module load openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export ALFWORLD_DATA="$DATA_ROOT/alfworld"
export TMPDIR="$ORION_ROOT/tmp/diagnostic-$SLURM_JOB_ID"
mkdir -p "$TMPDIR"
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$CODE_ROOT:$ORION_ROOT/code/webshop"
python -m "opd_pilot.$stage" --spec "$spec"
