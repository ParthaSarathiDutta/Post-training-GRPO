#!/usr/bin/env bash
# Run GRPO smoke on one GPU (intended for compute node or srun step).
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"
export CUDA_VISIBLE_DEVICES=0

LOG="${ROOT}/checkpoints/smoke/run.log"
mkdir -p "${ROOT}/checkpoints/smoke"

{
  echo "=== $(date -Is) ==="
  echo "hostname: $(hostname)"
  echo "SLURM_JOB_ID: ${SLURM_JOB_ID:-unset}"
  echo "SLURM_STEP_ID: ${SLURM_STEP_ID:-unset}"
  nvidia-smi -L || true
  python - <<'PY'
import torch
print("torch.cuda.is_available():", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
PY
  echo "--- starting smoke train ---"
  /usr/bin/time -f "elapsed_sec=%e max_rss_kb=%M" python train.py --smoke-test --output-dir checkpoints/smoke
  echo "--- outputs ---"
  find checkpoints/smoke -maxdepth 2 -type f 2>/dev/null | head -40
  ls -la checkpoints/smoke/ 2>/dev/null || true
} 2>&1 | tee "${LOG}"
