#!/usr/bin/env bash
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  JOBID="$(squeue -u "${USER}" -t RUNNING -h -o "%i" 2>/dev/null | head -1 || true)"
  if [[ -n "${JOBID}" ]]; then
    echo "Dispatching eval into existing allocation jobid=${JOBID}"
    exec srun --jobid="${JOBID}" --overlap --nodes=1 --ntasks=1 --cpus-per-task=32 --gpus=1 \
      bash "${ROOT}/scripts/run_eval_gpu.sh"
  fi
  echo "Starting 1-GPU srun for evaluation (account m3794, qos interactive)."
  exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --constraint=gpu \
    --account=m3794 --qos=interactive --time=02:00:00 --job-name=grpo_eval \
    --cpus-per-task=32 --gpus=1 \
    bash "${ROOT}/scripts/run_eval_gpu.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

LOG="${ROOT}/evaluation/eval_run.log"
mkdir -p "${ROOT}/evaluation"

echo "=== GRPO eval on $(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ===" | tee "${LOG}"
/usr/bin/time -f "elapsed_sec=%e max_rss_kb=%M" \
  python evaluate.py --num-examples 100 --batch-size 8 2>&1 | tee -a "${LOG}"
