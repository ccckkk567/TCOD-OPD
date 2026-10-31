#!/bin/bash
#SBATCH --job-name=opd-arrow-repair
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/env/arrow-%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Submit through Slurm}"
source orion/site.env
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0 openjdk/17.0.11
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -n 9 || { echo 'Shared environment still in use; no changes made'; exit 75; }
source "$ENV_ROOT/tcod-opd/bin/activate"
export PYTHONPATH="$PWD"
export TMPDIR="$ORION_ROOT/tmp/arrow-repair-$SLURM_JOB_ID"
export PIP_CACHE_DIR="$ORION_ROOT/downloads/pip-cache"
mkdir -p "$TMPDIR" "$LOG_ROOT/env/arrow-$SLURM_JOB_ID"
python -m pip freeze > "$LOG_ROOT/env/arrow-$SLURM_JOB_ID/before.txt"
python - <<'PY'
from importlib.metadata import distributions, version
from packaging.requirements import Requirement
import numpy
print({name: version(name) for name in ('numpy', 'pyarrow', 'datasets')}, flush=True)
if numpy.__version__ != '1.26.4':
    raise RuntimeError('Environment changed since reported failure; inspect before repair')
for package in distributions():
    for text in package.requires or []:
        req = Requirement(text)
        if req.name.lower() == 'pyarrow' and (req.marker is None or req.marker.evaluate()):
            if not req.specifier.contains('21.0.0'):
                raise RuntimeError(f'{package.metadata["Name"]} requires {req}; repair not applied')
PY
# Change only Arrow. Do not let the resolver upgrade NumPy or reinstall the training stack.
python -m pip install --only-binary=:all: --no-deps 'pyarrow==21.0.0'
python -m pip check
python - <<'PY'
import numpy as np
import pyarrow as pa
import datasets
import alfworld
import textworld
from trinity.buffer.reader.file_reader import _HFBatchReader
assert np.__version__ == '1.26.4'
assert pa.__version__ == '21.0.0'
array = np.array([1, 2, 3], dtype=np.int64)
assert pa.array(array).to_pylist() == [1, 2, 3]
data = datasets.Dataset.from_dict({'value': array})
assert list(data['value']) == [1, 2, 3]
print('ARROW_DATASETS_REPAIRED: imports and NumPy/Arrow/Dataset conversion passed; training not yet validated')
PY
python -m pip freeze > "$LOG_ROOT/env/arrow-$SLURM_JOB_ID/after.txt"
