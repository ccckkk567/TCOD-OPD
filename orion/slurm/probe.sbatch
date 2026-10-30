#!/bin/bash
# Resources and log paths are supplied by submit_probe.sh.
set -euo pipefail
: "${SLURM_JOB_ID:?Run this through sbatch}"
repo=${1:?Repository directory required}
kind=${2:?cpu or gpu required}
source "$repo/orion/site.env"
umask 077
mkdir -p "$CHECKPOINT_ROOT" "$OUTPUT_ROOT" "$REPORT_ROOT"

# Probes may fail individually: report their exit codes without hiding missing capabilities.
probe() {
    printf '\n--- %s ---\n' "$1"
    shift
    if "$@"; then :; else printf 'PROBE_UNAVAILABLE exit=%s\n' "$?"; fi
}
printf 'probe=%s job=%s host=%s\n' "$kind" "$SLURM_JOB_ID" "$(hostname)"
probe OS uname -a
probe RAM free -h
probe storage df -h "$ORION_ROOT" "$STORAGE_ROOT"
printf 'Filesystem capacity above is NOT a project quota.\n'
probe partitions sinfo -p testqueue,ioqueue
probe account sacctmgr -n show assoc user=linz3 format=Account,Partition,QOS
probe modules module -t avail
probe apptainer apptainer --version
probe git git --version
probe java java -version
probe containers find "$CONTAINER_ROOT" -maxdepth 2 -type f -name '*.sif' -printf '%p\n'
probe environments find "$ENV_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
probe models find "$MODEL_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
probe data find "$DATA_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n'
if [[ "$kind" == gpu ]]; then
    probe GPU nvidia-smi
elif [[ "$kind" != cpu ]]; then
    echo 'Unknown probe kind' >&2; exit 64
fi
for host in github.com huggingface.co api.swanlab.cn nvcr.io; do
    probe "HTTPS $host" curl --head --silent --show-error --connect-timeout 10 --max-time 20 "https://$host"
done
set +x
if [[ -r "$CREDENTIALS_FILE" ]]; then
    source "$CREDENTIALS_FILE"
    [[ -n "${HF_TOKEN:-}" ]] && echo 'HF_TOKEN=CONFIGURED' || echo 'HF_TOKEN=MISSING'
    [[ -n "${SWANLAB_API_KEY:-}" ]] && echo 'SWANLAB_API_KEY=CONFIGURED' || echo 'SWANLAB_API_KEY=MISSING'
else
    echo 'Credentials file unreadable'
fi
echo 'PROBE_FINISHED (capability report only; not an environment/training gate)'
