#!/usr/bin/env bash
# Start tmux grpo-full on THIS login node. Attach: tmux attach -t grpo-full
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
LOGIN_HOST="$(hostname -s)"

tmux kill-session -t grpo-full 2>/dev/null || true
# Remain-on-exit keeps pane output if salloc/train exits with an error.
tmux new-session -d -s grpo-full
tmux set-option -t grpo-full remain-on-exit on
tmux send-keys -t grpo-full "bash ${ROOT}/scripts/tmux_grpo_full.sh" C-m

echo "login_host=${LOGIN_HOST}"
echo "tmux attach -t grpo-full"
