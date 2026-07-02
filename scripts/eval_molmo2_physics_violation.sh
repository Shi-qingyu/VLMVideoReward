#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

export DATASET=${DATASET:-"physics_violation_sampled_test"}
export EVAL_ENTRY=${EVAL_ENTRY:-"${REPO_ROOT}/src/evaluation/evaluation_molmo2.py"}
export BACKEND=${BACKEND:-"hf"}
export OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/eval_results/single_issue/molmo2/${DATASET}"}

exec "${SCRIPT_DIR}/eval_physics_violation.sh"
