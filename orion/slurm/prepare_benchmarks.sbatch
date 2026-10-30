#!/bin/bash
#SBATCH --job-name=opd-bench-prep
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --array=0-2%1
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/benchmarks/prepare-%A_%a.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Run through sbatch}"
source orion/site.env
names=(alfworld webshop scienceworld)
name="${names[$SLURM_ARRAY_TASK_ID]}"
module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ORION_ROOT/downloads/pip-cache"
export PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=3
export TMPDIR="$ORION_ROOT/tmp/benchmark-$SLURM_JOB_ID"
metadata="$LOG_ROOT/benchmarks/$SLURM_JOB_ID"
mkdir -p "$TMPDIR" "$metadata" "$DATA_ROOT/$name" "$ENV_ROOT" "$ORION_ROOT/code"
# All three tasks mutate the same environment; serialize even across submissions.
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock 9
test -f "$ENV_ROOT/tcod-opd/ENV_INSTALLED.json"
source "$ENV_ROOT/tcod-opd/bin/activate"
cp -- "$0" "$metadata/submitted-script.sbatch"
git -C "$CODE_ROOT" rev-parse HEAD > "$metadata/asset-tools-revision.txt"
python -m pip freeze --exclude-editable > "$metadata/before.txt"
# Preserve the previously validated package set; conflicts must fail visibly.
export PIP_CONSTRAINT="$metadata/before.txt"
clone_once() {
    if [[ ! -e "$2" ]]; then
        git clone "$1" "$2"
    else
        test "$(git -C "$2" remote get-url origin)" = "$1"
        test -z "$(git -C "$2" status --porcelain --untracked-files=no)"
    fi
    git -C "$2" rev-parse HEAD > "$metadata/upstream-revision.txt"
}
case "$name" in
alfworld)
    python -m pip install alfworld
    export ALFWORLD_DATA="$DATA_ROOT/alfworld"
    alfworld-download --help > "$metadata/download-help.txt"
    alfworld-download --data-dir "$ALFWORLD_DATA"
    test -d "$ALFWORLD_DATA/json_2.1.1/train"
    split="$OUTPUT_ROOT/protocols/alfworld/pilot-v1"
    if [[ -f "$split/SPLITS_READY.json" ]]; then
        python -m opd_pilot.prepare --data-root "$ALFWORLD_DATA" --output "$split" --validate-only
    else
        python -m opd_pilot.prepare --data-root "$ALFWORLD_DATA" --output "$split"
    fi
    ;;
webshop)
    clone_once https://github.com/princeton-nlp/webshop.git "$ORION_ROOT/code/webshop"
    python -m pip install beautifulsoup4 cleantext Flask gdown 'gym==0.24.0' \
        'pyserini==0.17.0' rank_bm25 requests_mock scikit-learn selenium spacy thefuzz faiss-cpu
    for model in en_core_web_lg en_core_web_sm; do
        if ! python -c 'import importlib.util, sys; raise SystemExit(importlib.util.find_spec(sys.argv[1]) is None)' "$model"; then
            python -m spacy download "$model"
        fi
    done
    module load openjdk/17.0.11
    python orion/scripts/download_webshop.py "$DATA_ROOT/webshop"
    echo 'WebShop dependencies and small data prepared; search index and simulator validation remain pending.'
    ;;
scienceworld)
    module load openjdk/17.0.11
    clone_once https://github.com/allenai/ScienceWorld.git "$ORION_ROOT/code/ScienceWorld"
    python -m pip install "$ORION_ROOT/code/ScienceWorld"
    python - "$DATA_ROOT/scienceworld" <<'PY'
import importlib.metadata
import json
import shutil
import sys
from pathlib import Path
import scienceworld
source = Path(scienceworld.__file__).parent
output = Path(sys.argv[1])
for name in ('scienceworld.jar', 'tasks.json', 'object_type_ids.tsv'):
    target = output / name
    if target.exists():
        import filecmp
        if not filecmp.cmp(source / name, target, shallow=False):
            raise RuntimeError(f'Existing asset differs; inspect before replacing: {target}')
    else:
        shutil.copy2(source / name, target)
with (output / 'asset.json').open('w') as stream:
    json.dump({'version': importlib.metadata.version('scienceworld'),
               'simulator_validated': False}, stream, indent=2)
PY
    ;;
esac
python -m pip check
python "$CODE_ROOT/orion/scripts/check_env.py" "$metadata/gpu-check.json"
python -m pip freeze > "$metadata/requirements.txt"
printf 'Preparation completed; simulator/training readiness not certified. Metadata: %s\n' "$metadata" | tee "$ENV_ROOT/tcod-opd/BENCHMARK-$name.txt"
