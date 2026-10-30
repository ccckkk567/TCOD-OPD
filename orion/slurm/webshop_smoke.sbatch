#!/bin/bash
#SBATCH --job-name=opd-webshop-smoke
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/webshop-smoke-%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
# This stage may add the missing official spaCy model; exclude concurrent env users.
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock 9
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$CODE_ROOT:$ORION_ROOT/code/webshop"
export TMPDIR="$ORION_ROOT/tmp/webshop-smoke-$SLURM_JOB_ID"
metadata="$LOG_ROOT/benchmarks/webshop-smoke-$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$metadata"
python -m pip freeze --exclude-editable > "$metadata/before.txt"
export PIP_CONSTRAINT="$metadata/before.txt"
if ! python -c 'import importlib.util; raise SystemExit(importlib.util.find_spec("en_core_web_lg") is None)'; then
    python -m spacy download en_core_web_lg
fi
python -m pip check
python - "$metadata/index-spec.json" <<'PY'
import json, os, subprocess, sys
from pathlib import Path
repo = Path(os.environ['ORION_ROOT']) / 'code/webshop'
spec = dict(repo=str(repo), data_root=os.environ['DATA_ROOT'] + '/webshop',
            revision=subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True).strip())
Path(sys.argv[1]).write_text(json.dumps(spec, indent=2))
PY
python -m opd_pilot.webshop_index --spec "$metadata/index-spec.json"
python orion/scripts/benchmark_runtime_smoke.py webshop \
  --data-root "$DATA_ROOT/webshop" \
  --output "$REPORT_ROOT/benchmarks/webshop-smoke-$SLURM_JOB_ID"
python -m pip freeze > "$metadata/requirements.txt"
