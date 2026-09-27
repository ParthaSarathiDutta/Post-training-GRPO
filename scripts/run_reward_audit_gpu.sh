#!/usr/bin/env bash
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  JOBID="$(squeue -u "${USER}" -t RUNNING -h -o "%i" 2>/dev/null | head -1 || true)"
  if [[ -z "${JOBID}" ]]; then
    echo "No RUNNING allocation visible for ${USER}. Run this script from your salloc shell on nid..."
    exit 1
  fi
  exec srun --jobid="${JOBID}" --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=single:0 \
    bash "${ROOT}/scripts/run_reward_audit_gpu.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

python scripts/diagnose_completion_lengths.py
