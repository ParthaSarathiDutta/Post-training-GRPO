#!/usr/bin/env bash
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

if [[ "$(hostname)" == login* ]]; then
  echo "Starting 1-GPU srun for MATH-500 evaluation."
  exec srun --nodes=1 --ntasks=1 --gpus-per-task=1 --constraint=gpu \
    --account=m3794 --qos=interactive --time=02:00:00 --job-name=grpo_math500 \
    --cpus-per-task=32 --gpus=1 \
    bash "${ROOT}/scripts/run_math500_eval_gpu.sh"
fi

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

mkdir -p "${ROOT}/evaluation/math500"
LOG="${ROOT}/evaluation/math500/eval_run.log"
python evaluate_math500.py --batch-size 8 2>&1 | tee "${LOG}"
