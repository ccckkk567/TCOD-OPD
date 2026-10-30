#!/bin/bash
# Append the approved 50-update experiment to an existing successful recovery chain.
set -euo pipefail
root=/scratch/prjsnnn0000269/Students/shuming/cjk
repo="$root/code/TCOD-OPD"
revision=$(git -C "$repo" rev-parse "${1:?revision required}^{commit}")
resume=${2:?recovery job ID required}
[[ "$resume" =~ ^[0-9]+$ ]] || exit 2
record=$(sacct -X -n -P -j "$resume" --format=JobIDRaw,JobName%80,User,State)
matched=0
while IFS='|' read -r id name owner state rest; do
  if [[ "$id" == "$resume" && "$name" == opd-alf-s17-2g-resume && "$owner" == "$USER" ]]; then
    case "$state" in PENDING|RUNNING|CONFIGURING|COMPLETING|COMPLETED) matched=1 ;; esac
  fi
done <<< "$record"
[[ "$matched" == 1 ]] || { echo 'Expected an active/successful two-GPU recovery job owned by you'; exit 1; }
existing=$(squeue --noheader --user="$USER" --name=opd-alf-s17-2g-pilot --format='%i')
[[ -z "$existing" ]] || { echo "Pilot already queued/running: $existing"; exit 1; }
checkout="$root/tmp/alfworld-pilot-$(date +%Y%m%d-%H%M%S)"
git -C "$repo" worktree add --detach "$checkout" "$revision"
source "$checkout/orion/site.env"
test ! -e "$OUTPUT_ROOT/alfworld-s17-two-gpu-pilot/pilot/training-status.json"
mkdir -p "$LOG_ROOT/train"
job=$(sbatch --parsable --gres=gpu:2 --cpus-per-task=16 --mem=128G --time=18:00:00 \
  --dependency="afterok:$resume" --kill-on-invalid-dep=yes --job-name=opd-alf-s17-2g-pilot \
  --chdir="$checkout" "$checkout/orion/slurm/train.sbatch" \
  "$checkout/orion/generated/alfworld-s17-two-gpu-pilot/config.yaml" "$revision" "$resume")
printf 'pilot %s afterok:%s\n' "$job" "$resume" | tee "$checkout-jobs.txt"
