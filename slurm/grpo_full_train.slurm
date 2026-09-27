#!/bin/bash
#SBATCH --job-name=grpo_full
#SBATCH --account=m3794_g
#SBATCH --constraint=gpu
#SBATCH --qos=regular
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --output=/global/cfs/cdirs/m3560/Partha/GRPO/logs/grpo_full_%j.out
#SBATCH --error=/global/cfs/cdirs/m3560/Partha/GRPO/logs/grpo_full_%j.err

# Full GRPO + LoRA (train[:5%], 1 epoch). Submit from project root:
#   sbatch slurm/grpo_full_train.slurm

set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
cd "${ROOT}"
mkdir -p logs

module load pytorch/2.13.0
source "${ROOT}/.venv/bin/activate"
export HF_HOME="${SCRATCH}/hf_cache"

echo "=== GRPO full training ==="
echo "JobID=${SLURM_JOB_ID} Node=$(hostname) Start=$(date -Is)"
nvidia-smi -L || true

python train.py --output-dir checkpoints/Qwen2-0.5B-GRPO

echo "=== Finished $(date -Is) ==="
