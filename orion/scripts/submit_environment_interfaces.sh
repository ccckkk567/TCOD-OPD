#!/bin/bash
# Lightweight login-node submission. No checkout changes, installs or new clones.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision="${1:?Pass the fetched Git revision}"
revision=$(git -C "$repo" rev-parse "${revision}^{commit}")
snapshot="$root/tmp/environment-interface-$(date +%Y%m%d-%H%M%S)"
mkdir "$snapshot"
git -C "$repo" archive --format=tar --output="$snapshot/source.tar" "$revision" \
  opd_pilot/__init__.py opd_pilot/environment.py orion/site.env \
  orion/scripts/environment_interface_smoke.py orion/slurm/environment_interface_smoke.sbatch \
  trinity/common/workflows/envs/TCOD/alfworld/utils.py \
  trinity/common/workflows/envs/TCOD/scienceworld/utils.py
tar -xf "$snapshot/source.tar" -C "$snapshot"
printf '%s\n' "$revision" > "$snapshot/revision.txt"
mkdir -p "$root/logs/tcod-opd/benchmarks"
# Use full job names and expanded array IDs. Wait for queued as well as running writers.
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
for benchmark in alfworld scienceworld; do
  job=$(sbatch --parsable "${options[@]}" --job-name="opd-$benchmark-interface" \
    --chdir="$snapshot" "$snapshot/orion/slurm/environment_interface_smoke.sbatch" "$benchmark")
  printf '%s %s\n' "$benchmark" "$job" | tee -a "$snapshot/jobs.txt"
done
printf 'Script snapshot (not a clone): %s\n' "$snapshot"
