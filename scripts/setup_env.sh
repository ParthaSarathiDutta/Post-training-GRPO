#!/usr/bin/env bash
# Perlmutter: venv with module pytorch + pip extras for GRPO.
set -euo pipefail

ROOT="/global/cfs/cdirs/m3560/Partha/GRPO"
VENV_DIR="${VENV_DIR:-${ROOT}/.venv}"

module load pytorch/2.13.0

if [[ ! -d "${VENV_DIR}" ]]; then
  python3 -m venv --system-site-packages "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade pip
pip install -r "${ROOT}/requirements.txt"

export HF_HOME="${HF_HOME:-${SCRATCH:-${ROOT}/.cache}/hf_cache}"
mkdir -p "${HF_HOME}"

echo "Ready: source ${VENV_DIR}/bin/activate"
echo "HF_HOME=${HF_HOME}"
