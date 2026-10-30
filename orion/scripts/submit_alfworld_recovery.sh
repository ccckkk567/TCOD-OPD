#!/bin/bash
# Prepared only: invoke after the user confirms eight GPUs for native training.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision=$(git -C "$repo" rev-parse "${1:?revision required}^{commit}")
[[ "${2:-}" == --approved-eight-gpu ]] || { echo 'Requires explicit eight-GPU approval'; exit 2; }
checkout="$root/tmp/alfworld-recovery-$(date +%Y%m%d-%H%M%S)"
git -C "$repo" worktree add --detach "$checkout" "$revision"
source "$checkout/orion/site.env"
for name in Qwen3-1.7B Qwen3-8B; do test -f "$MODEL_ROOT/$name/READY.json"; done
test -f "$OUTPUT_ROOT/protocols/alfworld/pilot-v1/train_pilot.jsonl"
test ! -e "$OUTPUT_ROOT/alfworld-s17-smoke/smoke-first/training-status.json"
test ! -e "$OUTPUT_ROOT/alfworld-s17-smoke/smoke-resume/training-status.json"
mkdir -p "$LOG_ROOT/train"
options=()
deps=()
queue=$(squeue --noheader --array --user="$USER" --format='%i|%j')
while IFS='|' read -r job name; do
  case "$name" in
    opd-install|opd-bench-prep|opd-textworld-meta|opd-webshop-smoke) deps+=("$job") ;;
    opd-alf-s17-first|opd-alf-s17-resume) echo 'Recovery chain already queued/running'; exit 1 ;;
  esac
done <<< "$queue"
if ((${#deps[@]})); then
  joined=$(IFS=:; echo "${deps[*]}")
  options+=(--dependency="afterany:$joined")
fi
first=$(sbatch --parsable "${options[@]}" --job-name=opd-alf-s17-first --chdir="$checkout" \
  "$checkout/orion/slurm/train.sbatch" \
  "$checkout/orion/generated/alfworld-s17-smoke-first/config.yaml" "$revision")
first=${first%%;*}
printf 'smoke-first %s\n' "$first" | tee "$checkout-jobs.txt"
resume=$(sbatch --parsable --dependency="afterok:$first" --kill-on-invalid-dep=yes \
  --job-name=opd-alf-s17-resume --chdir="$checkout" "$checkout/orion/slurm/train.sbatch" \
  "$checkout/orion/generated/alfworld-s17-smoke-resume/config.yaml" "$revision")
printf 'smoke-resume %s\n' "$resume" | tee -a "$checkout-jobs.txt"
