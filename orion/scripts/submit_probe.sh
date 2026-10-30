#!/bin/bash
set -euo pipefail
repo=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
source "$repo/orion/site.env"
mkdir -p "$LOG_ROOT/probe"
for kind in cpu gpu; do
    args=(--parsable --account="$ACCOUNT" --nodes=1 --ntasks=1
          --cpus-per-task=2 --mem=8G --time=00:15:00
          --job-name="opd-probe-$kind" --chdir="$repo"
          --output="$LOG_ROOT/probe/$kind-%j.log")
    if [[ "$kind" == gpu ]]; then
        args+=(--partition=testqueue --gres=gpu:1)
    else
        args+=(--partition=ioqueue)
    fi
    job=$(sbatch "${args[@]}" "$repo/orion/slurm/probe.sbatch" "$repo" "$kind")
    printf '%s\t%s\n' "$kind" "$job" | tee -a "$LOG_ROOT/probe/jobs.tsv"
done
