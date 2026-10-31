#!/bin/bash
# Preserve failed runs and unrelated jobs. Repair first, then run the full approved chain.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision=$(git -C "$repo" rev-parse "${1:?revision required}^{commit}")
queue=$(squeue --noheader --array --user="$USER" --format='%i|%j')
while IFS='|' read -r job name; do
  case "$name" in opd-*) echo "Project job still queued/running: $job $name; wait before changing shared environment"; exit 1 ;; esac
done <<< "$queue"
checkout="$root/tmp/alfworld-arrow-repair-$(date +%Y%m%d-%H%M%S)"
git -C "$repo" worktree add --detach "$checkout" "$revision"
source "$checkout/orion/site.env"
for run in alfworld-s17-two-gpu-smoke-arrow-fix alfworld-s17-two-gpu-pilot-arrow-fix; do
  test ! -e "$OUTPUT_ROOT/$run" || { echo "Run already exists: $run"; exit 1; }
  test ! -e "$CHECKPOINT_ROOT/$run" || { echo "Checkpoint run already exists: $run"; exit 1; }
done
for model in Qwen3-1.7B Qwen3-8B; do test -f "$MODEL_ROOT/$model/READY.json"; done
mkdir -p "$LOG_ROOT/env" "$LOG_ROOT/train"
registry="$checkout-jobs.txt"
repair=$(sbatch --parsable --chdir="$checkout" "$checkout/orion/slurm/repair_arrow.sbatch")
repair=${repair%%;*}
printf 'repair %s\n' "$repair" | tee "$registry"
common=(--parsable --gres=gpu:2 --cpus-per-task=16 --mem=128G --chdir="$checkout" --kill-on-invalid-dep=yes)
first=$(sbatch "${common[@]}" --dependency="afterok:$repair" --job-name=opd-alf-s17-2g-first \
  "$checkout/orion/slurm/train.sbatch" "$checkout/orion/generated/alfworld-s17-arrow-fix-smoke-first/config.yaml" "$revision")
first=${first%%;*}
printf 'smoke-first %s\n' "$first" | tee -a "$registry"
resume=$(sbatch "${common[@]}" --dependency="afterok:$first" --job-name=opd-alf-s17-2g-resume \
  "$checkout/orion/slurm/train.sbatch" "$checkout/orion/generated/alfworld-s17-arrow-fix-smoke-resume/config.yaml" "$revision")
resume=${resume%%;*}
printf 'smoke-resume %s\n' "$resume" | tee -a "$registry"
pilot=$(sbatch "${common[@]}" --dependency="afterok:$resume" --job-name=opd-alf-s17-2g-pilot --time=18:00:00 \
  "$checkout/orion/slurm/train.sbatch" "$checkout/orion/generated/alfworld-s17-arrow-fix-pilot/config.yaml" \
  "$revision" "$resume" "$OUTPUT_ROOT/alfworld-s17-two-gpu-smoke-arrow-fix")
printf 'pilot %s\n' "$pilot" | tee -a "$registry"
