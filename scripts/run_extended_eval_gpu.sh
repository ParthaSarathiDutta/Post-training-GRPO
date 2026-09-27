#!/usr/bin/env bash
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  echo "Starting 1-GPU srun for extended evaluation (GSM8K + checkpoint curve)."
  exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --constraint=gpu \
    --account=m3794 --qos=interactive --time=04:00:00 --job-name=grpo_ext_eval \
    --cpus-per-task=32 --gpus=1 \
    bash "${ROOT}/scripts/run_extended_eval_gpu.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

LOG="${ROOT}/evaluation/extended_eval_run.log"
mkdir -p "${ROOT}/evaluation/gsm8k" "${ROOT}/evaluation/checkpoints" "${ROOT}/evaluation/plots"

{
  echo "=== Extended eval on $(hostname) $(date -Is) ==="
  python evaluate_gsm8k.py --num-examples 500 --batch-size 8
  python evaluate_checkpoints.py --num-examples 100 --batch-size 8
} 2>&1 | tee "${LOG}"
