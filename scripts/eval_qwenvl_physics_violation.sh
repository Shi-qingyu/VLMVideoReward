#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

export DATASET=${DATASET:-"physics_violation_sampled_test"}
export EVAL_ENTRY=${EVAL_ENTRY:-"${REPO_ROOT}/src/evaluation/evaluation_qwen.py"}
export BACKEND=${BACKEND:-"vllm"}
export OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/eval_results/single_issue/qwenvl/${DATASET}"}

exec "${SCRIPT_DIR}/eval_physics_violation.sh"
