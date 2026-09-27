#!/usr/bin/env bash
# Full GRPO training on one GPU inside the current salloc allocation.
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
LOG="${ROOT}/logs/grpo_full_interactive.log"

run_train() {
  mkdir -p "${ROOT}/logs"
  cd "${ROOT}"
  module load pytorch/2.13.0
  source "${ROOT}/.venv/bin/activate"
  export HF_HOME="${SCRATCH}/hf_cache"
  export CUDA_VISIBLE_DEVICES=0

  {
    echo "=== GRPO full (interactive/tmux) ==="
    echo "hostname=$(hostname) slurm_job=${SLURM_JOB_ID:-unset} step=${SLURM_STEP_ID:-unset} start=$(date -Is)"
    nvidia-smi -L || true
    python -c "import torch; print('cuda', torch.cuda.is_available(), 'count', torch.cuda.device_count()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a')"
  } | tee -a "${LOG}"

  python train.py --output-dir checkpoints/Qwen2-0.5B-GRPO 2>&1 | tee -a "${LOG}"
  echo "=== Finished $(date -Is) ===" | tee -a "${LOG}"
}

if [[ "$(hostname)" == login* ]]; then
  echo "Launching 1-GPU srun step in allocation ${SLURM_JOB_ID}..." | tee -a "${LOG}"
  exec srun --jobid="${SLURM_JOB_ID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
    bash "${ROOT}/scripts/run_full_train_logged.sh"
fi

run_train
