#!/usr/bin/env bash
set -euo pipefail

ROOT="${CELAR_ROOT:-/home/scu101/celar/attm5}"
PYTHON_BIN="${CELAR_PYTHON:-python3}"

cd "${ROOT}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

"${PYTHON_BIN}" -m pip install -r requirements-sim.txt
"${PYTHON_BIN}" -m pytest -q
"${PYTHON_BIN}" -m celar_sim.cli --config configs/c0_smoke.yaml
"${PYTHON_BIN}" -m celar_sim.cli --config configs/c0_mujoco.yaml
