#!/bin/bash
# Two independent one-GPU jobs. Immutable source snapshot, no clone or checkout changes.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision="${1:?Pass the fetched Git revision}"
shift
benchmarks=("$@")
if ((${#benchmarks[@]} == 0)); then benchmarks=(alfworld scienceworld); fi
for benchmark in "${benchmarks[@]}"; do
  case "$benchmark" in alfworld|scienceworld) ;; *) echo "Unknown benchmark: $benchmark"; exit 2 ;; esac
done
revision=$(git -C "$repo" rev-parse "${revision}^{commit}")
for model in Qwen3-1.7B Qwen3-8B; do
  test -f "$root/models/$model/READY.json" || { echo "Missing completed model: $model"; exit 1; }
done
test -f "$root/data/scienceworld/scienceworld.jar"
test -f /scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/outputs/tcod-opd/protocols/alfworld/pilot-v1/smoke.jsonl
snapshot="$root/tmp/model-interface-$(date +%Y%m%d-%H%M%S)"
mkdir "$snapshot"
git -C "$repo" archive --format=tar --output="$snapshot/source.tar" "$revision" \
  opd_pilot orion/site.env orion/scripts/environment_interface_smoke.py \
  orion/scripts/model_environment_smoke.py orion/slurm/model_environment_smoke.sbatch \
  trinity/common/workflows/envs/TCOD/alfworld/utils.py \
  trinity/common/workflows/envs/TCOD/scienceworld/utils.py
tar -xf "$snapshot/source.tar" -C "$snapshot"
printf '%s\n' "$revision" > "$snapshot/revision.txt"
mkdir -p "$root/logs/tcod-opd/model-interface"
queue=$(squeue --noheader --array --user="$USER" --format='%i|%j')
dependencies=()
while IFS='|' read -r job name; do
  case "$name" in
    opd-install|opd-bench-prep|opd-textworld-meta|opd-webshop-smoke)
      dependencies+=("$job") ;;
  esac
done <<< "$queue"
options=()
if ((${#dependencies[@]})); then
  joined=$(IFS=:; echo "${dependencies[*]}")
  options+=(--dependency="afterany:$joined")
  printf 'Waiting for shared-environment jobs: %s\n' "$joined"
fi
for benchmark in "${benchmarks[@]}"; do
  job=$(sbatch --parsable "${options[@]}" --job-name="opd-$benchmark-model" \
    --chdir="$snapshot" "$snapshot/orion/slurm/model_environment_smoke.sbatch" "$benchmark")
  printf '%s %s\n' "$benchmark" "$job" | tee -a "$snapshot/jobs.txt"
done
printf 'Source snapshot (not a clone): %s\n' "$snapshot"
