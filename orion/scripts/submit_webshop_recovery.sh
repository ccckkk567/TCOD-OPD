#!/bin/bash
# Reuse existing assets, leave active checkouts unchanged, serialize environment writes.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision="${1:?Pass fetched Git revision}"
revision=$(git -C "$repo" rev-parse "${revision}^{commit}")
snapshot="$root/tmp/webshop-recovery-$(date +%Y%m%d-%H%M%S)"
mkdir "$snapshot"
git -C "$repo" archive --format=tar --output="$snapshot/source.tar" "$revision" \
  orion/site.env orion/slurm/webshop_smoke.sbatch orion/scripts/benchmark_runtime_smoke.py \
  opd_pilot/__init__.py opd_pilot/webshop_index.py \
  trinity/common/workflows/envs/TCOD/webshop/utils.py
tar -xf "$snapshot/source.tar" -C "$snapshot"
printf '%s\n' "$revision" > "$snapshot/revision.txt"
mkdir -p "$root/logs/tcod-opd/benchmarks"
queue=$(squeue --noheader --array --user="$USER" --format='%i|%j')
dependencies=()
while IFS='|' read -r job name; do
  case "$name" in opd-*) dependencies+=("$job") ;; esac
done <<< "$queue"
options=()
if ((${#dependencies[@]})); then
  joined=$(IFS=:; echo "${dependencies[*]}")
  options+=(--dependency="afterany:$joined")
  printf 'Waiting for existing OPD environment users: %s\n' "$joined"
fi
job=$(sbatch --parsable "${options[@]}" --export=ALL,OPD_WEBSHOP_REUSE_ONLY=1 \
  --chdir="$snapshot" "$snapshot/orion/slurm/webshop_smoke.sbatch")
printf 'webshop-recovery %s\n' "$job" | tee "$snapshot/jobs.txt"
printf 'Log: %s/logs/tcod-opd/benchmarks/webshop-smoke-%s.log\n' "$root" "${job%%;*}"
