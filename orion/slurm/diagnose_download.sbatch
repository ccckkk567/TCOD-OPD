#!/bin/bash
#SBATCH --job-name=opd-download-check
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=ioqueue
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/models/diagnose-%j.log
set -uo pipefail
set +x
ulimit -c 0
root=/scratch/prjsnnn0000269/Students/shuming/cjk
printf 'job=%s host=%s\n' "$SLURM_JOB_ID" "$(hostname)"
lscpu
module purge
module load python/3.11.9 || exit 1
unset PYTHONHOME PYTHONPATH HF_TOKEN SWANLAB_API_KEY
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
probe() {
    printf '\nCHECK: %s\n' "$1"
    shift
    "$@"
    printf 'CHECK_EXIT=%s\n' "$?"
}
probe module-python python -X faulthandler -c 'import sys, ssl; print(sys.version); print(ssl.OPENSSL_VERSION)'
py="$root/envs/tcod-opd/bin/python"
probe venv-python "$py" -X faulthandler -c 'import sys, ssl; print(sys.version); print(ssl.OPENSSL_VERSION)'
probe versions "$py" -X faulthandler -c 'import importlib.metadata as m; print({n:m.version(n) for n in ("huggingface-hub", "hf-xet", "requests", "urllib3")})'
for package in requests huggingface_hub hf_xet; do
    probe "import-$package" "$py" -X faulthandler -c "import importlib; importlib.import_module('$package'); print('IMPORT_OK')"
done
probe hub-api "$py" -X faulthandler -c 'from huggingface_hub import HfApi, snapshot_download; HfApi(); print("API_INIT_OK")'
cd "$root/code/TCOD-OPD" || exit 1
probe assets-cli "$py" -X faulthandler -m opd_pilot.assets --help
echo 'DIAGNOSTIC_FINISHED: inspect individual CHECK_EXIT values; no download attempted.'
