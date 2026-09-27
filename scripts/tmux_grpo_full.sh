#!/usr/bin/env bash
# Runs inside tmux session grpo-full on a Perlmutter login node.
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
mkdir -p "${ROOT}/logs"
cd "${ROOT}"

echo "Waiting for salloc (exact command)..." | tee "${ROOT}/logs/grpo_full_interactive.log"

# Exact salloc only; training launcher runs inside the allocation.
salloc --nodes 2 --qos interactive --time 04:00:00 --ntasks-per-node=4 --constraint gpu --gpus-per-task=1 --gpus 8 --gpu-bind=none --account m3794 \
  bash "${ROOT}/scripts/run_full_train_logged.sh"
