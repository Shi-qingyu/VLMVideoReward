#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

INPUT=${INPUT:-"${REPO_ROOT}/data/physics_violation_sampled_train.json"}
OUTPUT=${OUTPUT:-"${REPO_ROOT}/data/physics_violation_sampled_train_qwen3vl4b_captioned.json"}
MODEL_PATH=${MODEL_PATH:-"Qwen/Qwen3-VL-4B-Instruct"}
BACKEND=${BACKEND:-"vllm"}
NUM_GPUS=${NUM_GPUS:-1}
BATCH_SIZE=${BATCH_SIZE:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-64}
VIDEO_FPS=${VIDEO_FPS:-2}
VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:-20}
VIDEO_MIN_PIXELS=${VIDEO_MIN_PIXELS:-1048576}
VIDEO_MAX_PIXELS=${VIDEO_MAX_PIXELS:-4194304}
WRITE_EVERY=${WRITE_EVERY:-20}
RESUME=${RESUME:-1}

cmd=(
    python3 "${REPO_ROOT}/tools/annotate_video_captions_qwen.py"
    --input "${INPUT}"
    --output "${OUTPUT}"
    --model-path "${MODEL_PATH}"
    --model-type qwen3vl
    --backend "${BACKEND}"
    --num-gpus "${NUM_GPUS}"
    --batch-size "${BATCH_SIZE}"
    --max-new-tokens "${MAX_NEW_TOKENS}"
    --video-fps "${VIDEO_FPS}"
    --video-max-frames "${VIDEO_MAX_FRAMES}"
    --video-min-pixels "${VIDEO_MIN_PIXELS}"
    --video-max-pixels "${VIDEO_MAX_PIXELS}"
    --write-every "${WRITE_EVERY}"
)

if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
    cmd+=(--allowed-local-media-path "${ALLOWED_LOCAL_MEDIA_PATH}")
fi

for arg in "$@"; do
    if [[ "${arg}" == "--force" ]]; then
        RESUME=0
    fi
done

if [[ "${RESUME}" != "0" ]]; then
    cmd+=(--resume)
fi

cd "${REPO_ROOT}"
"${cmd[@]}" "$@"
