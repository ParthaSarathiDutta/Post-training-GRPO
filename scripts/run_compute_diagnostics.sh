#!/usr/bin/env bash
# GPU diagnostics + smoke rerun on compute node only.
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  JOBID="$(squeue -u "${USER}" -t RUNNING -h -o "%i" 2>/dev/null | head -1 || true)"
  if [[ -z "${JOBID}" ]]; then
    echo "On login node and no RUNNING Slurm job for ${USER}."
    echo "Run from your salloc shell on nid..., or: bash ${ROOT}/scripts/run_smoke_from_login.sh"
    exit 1
  fi
  exec srun --jobid="${JOBID}" --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=single:0 \
    bash "${ROOT}/scripts/run_compute_diagnostics.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

echo "=== compute diagnostics on $(hostname) ==="
python scripts/verify_grpo_batch_config.py
python scripts/diagnose_smoke_generations.py
echo "=== smoke rerun ==="
bash "${ROOT}/scripts/run_smoke_gpu.sh"
