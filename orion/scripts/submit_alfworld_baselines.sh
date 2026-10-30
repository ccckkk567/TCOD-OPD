#!/bin/bash
# Login-node lightweight operations only. No installs and no model/data copying.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision=$(git -C "$repo" rev-parse "${1:?revision required}^{commit}")
stamp=$(date +%Y%m%d-%H%M%S)
checkout="$root/tmp/alfworld-experiment-$stamp"
# A detached worktree shares the existing Git objects; it is not another clone.
git -C "$repo" worktree add --detach "$checkout" "$revision"
bundle="/scratch/prjsnnn0000269/Students/shuming/storage_ext/cjk/outputs/tcod-opd/alfworld-baseline-$stamp"
logs="$root/logs/tcod-opd/alfworld-baseline-$stamp"
mkdir -p "$logs"
for name in Qwen3-1.7B Qwen3-8B; do
  test -f "$root/models/$name/READY.json" || { echo "Model not ready: $name"; exit 1; }
done
options=()
deps=()
queue=$(squeue --noheader --array --user="$USER" --format='%i|%j')
while IFS='|' read -r job name; do
  case "$name" in
    opd-install|opd-bench-prep|opd-textworld-meta|opd-webshop-smoke) deps+=("$job") ;;
  esac
done <<< "$queue"
if ((${#deps[@]})); then
  joined=$(IFS=:; echo "${deps[*]}")
  options+=(--dependency="afterany:$joined")
fi
script="$checkout/orion/slurm/alfworld_baseline.sbatch"
prep=$(sbatch --parsable "${options[@]}" --job-name=opd-alf-baseline-prep \
  --chdir="$checkout" --time=00:05:00 --output="$logs/prepare-%j.log" "$script" prepare "$bundle")
prep=${prep%%;*}
printf 'prepare %s\n' "$prep" | tee "$logs/jobs.txt"
for model in s17 t08; do
  job=$(sbatch --parsable --dependency="afterok:$prep" --kill-on-invalid-dep=yes \
    --job-name="opd-alf-baseline-$model" --chdir="$checkout" --output="$logs/$model-%j.log" \
    "$script" evaluate "$bundle" "$model")
  printf '%s %s\n' "$model" "$job" | tee -a "$logs/jobs.txt"
done
printf 'checkout=%s\nbundle=%s\nlogs=%s\n' "$checkout" "$bundle" "$logs"
