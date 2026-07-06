#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

INPUT=${INPUT:-"${REPO_ROOT}/data/physics_violation_sampled_train.json"}
OUTPUT=${OUTPUT:-"${REPO_ROOT}/data/physics_violation_sampled_train_sam3_time_bbox.json"}
WORK_DIR=${WORK_DIR:-"${REPO_ROOT}/data/sam3_time_bbox_shards"}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-12358}
PROMPT_MODE=${PROMPT_MODE:-heuristic}
MIN_SCORE=${MIN_SCORE:-0.0}
WRITE_EVERY=${WRITE_EVERY:-20}
RESUME=${RESUME:-1}

common_args=(
    "${REPO_ROOT}/tools/annotate_time_bbox_sam3.py"
    --input "${INPUT}"
    --output "${OUTPUT}"
    --work-dir "${WORK_DIR}"
    --prompt-mode "${PROMPT_MODE}"
    --min-score "${MIN_SCORE}"
    --write-every "${WRITE_EVERY}"
)

for arg in "$@"; do
    if [[ "${arg}" == "--force" ]]; then
        RESUME=0
    fi
done

if [[ "${RESUME}" != "0" ]]; then
    common_args+=(--resume)
fi

if [[ "${NPROC_PER_NODE}" -gt 1 ]]; then
    cd "${REPO_ROOT}"
    torchrun \
        --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${MASTER_PORT}" \
        "${common_args[@]}" "$@"
else
    cd "${REPO_ROOT}"
    python3 "${common_args[@]}" "$@"
fi
