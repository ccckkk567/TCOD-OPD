#!/bin/bash
#SBATCH --job-name=opd-install
#SBATCH --account=prjsnnn0000269
#SBATCH --partition=testqueue
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --chdir=/scratch/prjsnnn0000269/Students/shuming/cjk/code/TCOD-OPD
#SBATCH --output=/scratch/prjsnnn0000269/Students/shuming/cjk/logs/tcod-opd/env/install-%j.log
set -euo pipefail
set +x
: "${SLURM_JOB_ID:?Run through sbatch}"
source orion/site.env
umask 077
env_dir="$ENV_ROOT/tcod-opd"
metadata="$LOG_ROOT/env/$SLURM_JOB_ID"
mkdir -p "$ENV_ROOT" "$metadata" "$ORION_ROOT/tmp" "$ORION_ROOT/downloads/pip-cache"
exec 9>"$ENV_ROOT/tcod-opd.install.lock"
flock -n 9 || { echo 'Another installation is running'; exit 1; }
if [[ -f "$env_dir/ENV_INSTALLED.json" ]]; then
    echo 'Environment already installed; refusing to change a published environment.'
    exit 0
fi
trap 'rc=$?; printf "Install job finished: exit=%s\n" "$rc"' EXIT

module purge
module load python/3.11.9 cuda/12.6.2 gcc/10.5.0
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$ORION_ROOT/downloads/pip-cache"
export PIP_DEFAULT_TIMEOUT=60 PIP_RETRIES=3
export TMPDIR="$ORION_ROOT/tmp/tcod-install-$SLURM_JOB_ID"
export MAX_JOBS=4 OMP_NUM_THREADS=4
export TORCH_CUDA_ARCH_LIST=9.0
export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
mkdir -p "$TMPDIR"
[[ -x "$env_dir/bin/python" ]] || python -m venv "$env_dir"
source "$env_dir/bin/activate"
python --version
nvcc --version
nvidia-smi
git rev-parse HEAD > "$metadata/code-revision.txt"
module -t list 2> "$metadata/modules.txt"
python -m pip install --upgrade pip setuptools wheel packaging ninja psutil

# README editable install, with the vLLM extra required by the OPD configs.
# Pin the CUDA wheel family instead of letting a later resolver switch Torch.
python -m pip install -c orion/constraints.txt \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
python -m pip install --dry-run --report "$metadata/resolution.json" \
    -c orion/constraints.txt -e '.[dev,vllm]' swanlab huggingface_hub
python -m pip install -c orion/constraints.txt -e '.[dev,vllm]' swanlab huggingface_hub

# FlashAttention 2.8.1 has no published cp311/Torch 2.8 wheel. Build on this GPU.
# Site compiler is CUDA 12.6; Torch/vLLM runtime is CUDA 12.8. Record this minor
# version difference, keep compiler warnings, and test the resulting kernel.
export FLASH_ATTENTION_FORCE_BUILD=TRUE
python -m pip install 'flash-attn==2.8.1' --no-build-isolation --no-deps
python -m pip check
python -m pip freeze > "$metadata/requirements.lock.txt"
python orion/scripts/check_env.py "$metadata/gpu-check.json"
python -m trinity.cli.launcher --help
cp "$metadata/requirements.lock.txt" "$env_dir/requirements.lock.txt"
cp "$metadata/gpu-check.json" "$env_dir/ENV_INSTALLED.json"
echo 'ENV_INSTALLED: imports and attention forward/backward passed; model/OPD checks still pending.'
