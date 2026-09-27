#!/usr/bin/env bash
# Dispatch smoke to an existing salloc via srun (do not submit a new allocation).
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
JOBID="$(squeue -u "${USER}" -t RUNNING -h -o "%i" 2>/dev/null | head -1 || true)"

if [[ -z "${JOBID}" ]]; then
  echo "No RUNNING Slurm job for ${USER}. Active salloc required."
  echo "From your salloc shell, run: bash ${ROOT}/scripts/run_smoke_gpu.sh"
  exit 1
fi

echo "Using existing allocation jobid=${JOBID}"
exec srun --jobid="${JOBID}" --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=single:0 \
  bash "${ROOT}/scripts/run_smoke_gpu.sh"
