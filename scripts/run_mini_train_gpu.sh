#!/usr/bin/env bash
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  JOBID="$(squeue -u "${USER}" -t RUNNING -h -o "%i" 2>/dev/null | head -1 || true)"
  if [[ -n "${JOBID}" ]]; then
    echo "Dispatching mini-train into existing allocation jobid=${JOBID}"
    exec srun --jobid="${JOBID}" --nodes=1 --ntasks=1 --gpus-per-task=1 --gpu-bind=single:0 \
      bash "${ROOT}/scripts/run_mini_train_gpu.sh"
  fi
  echo "No RUNNING job in squeue; starting 1-GPU interactive srun step (account m3794, qos interactive)."
  exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --constraint=gpu \
    --account=m3794 --qos=interactive --time=04:00:00 --job-name=grpo_mini \
    --gpu-bind=single:0 \
    bash "${ROOT}/scripts/run_mini_train_gpu.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

LOG="${ROOT}/checkpoints/mini/mini_train.log"
mkdir -p "${ROOT}/checkpoints/mini"

echo "=== mini-train on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
/usr/bin/time -f "elapsed_sec=%e max_rss_kb=%M" \
  python train.py --mini-train --output-dir checkpoints/mini 2>&1 | tee "${LOG}"

python scripts/report_mini_train.py 2>&1 | tee -a "${LOG}"
