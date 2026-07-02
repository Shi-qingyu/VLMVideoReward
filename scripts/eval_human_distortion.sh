#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH=/path/to/checkpoint or MODEL_PATH=hf/model-id}

DATASET=${DATASET:-"human_distortion_sampled_test"}
EVAL_ENTRY=${EVAL_ENTRY:-"${REPO_ROOT}/src/evaluation/evaluation_qwen.py"}
MODEL_TYPE=${MODEL_TYPE:-auto}
BACKEND=${BACKEND:-vllm}
OUTPUT_DIR=${OUTPUT_DIR:-"${REPO_ROOT}/eval_results/single_issue/${DATASET}"}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-4}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
MODEL_MAX_LENGTH=${MODEL_MAX_LENGTH:-8192}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-8}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.7}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
ALLOWED_LOCAL_MEDIA_PATH=${ALLOWED_LOCAL_MEDIA_PATH:-"/mnt/bn/foundation-ads/"}
VIDEO_MAX_FRAMES=${VIDEO_MAX_FRAMES:-8}
VIDEO_FPS=${VIDEO_FPS:-1.0}

RUN_JUDGE=${RUN_JUDGE:-true}
JUDGE_MODEL=${JUDGE_MODEL:-"Qwen/Qwen3-30B-A3B-Instruct-2507"}
JUDGE_BATCH_SIZE=${JUDGE_BATCH_SIZE:-32}
JUDGE_MAX_NEW_TOKENS=${JUDGE_MAX_NEW_TOKENS:-256}
JUDGE_TENSOR_PARALLEL_SIZE=${JUDGE_TENSOR_PARALLEL_SIZE:-${TENSOR_PARALLEL_SIZE}}
JUDGE_GPU_MEMORY_UTILIZATION=${JUDGE_GPU_MEMORY_UTILIZATION:-0.75}
JUDGE_MAX_MODEL_LEN=${JUDGE_MAX_MODEL_LEN:-8192}

mkdir -p "${OUTPUT_DIR}"

python "${EVAL_ENTRY}" \
    --model_path "${MODEL_PATH}" \
    --model_type "${MODEL_TYPE}" \
    --backend "${BACKEND}" \
    --dataset_use "${DATASET}" \
    --output_dir "${OUTPUT_DIR}" \
    --metric_schema single_issue \
    --batch_size "${EVAL_BATCH_SIZE}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --model_max_length "${MODEL_MAX_LENGTH}" \
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}" \
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION}" \
    --max_num_seqs "${MAX_NUM_SEQS}" \
    --allowed_local_media_path "${ALLOWED_LOCAL_MEDIA_PATH}" \
    --video_max_frames "${VIDEO_MAX_FRAMES}" \
    --video_fps "${VIDEO_FPS}"

result_json=$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name '*.json' \
    ! -name '*_metrics.json' \
    ! -name '*_judgment.json' \
    ! -name '*_reasoning_consistency.json' \
    | sort | head -n 1)

if [[ -z "${result_json}" ]]; then
    echo "No evaluation result JSON found in ${OUTPUT_DIR}" >&2
    exit 1
fi

if [[ "${RUN_JUDGE}" == "true" ]]; then
    python tools/qwen_judge.py \
        --model_name_or_path "${JUDGE_MODEL}" \
        --input_file "${result_json}" \
        --output_file "${result_json%.json}_reasoning_consistency.json" \
        --batch_size "${JUDGE_BATCH_SIZE}" \
        --max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
        --tensor_parallel_size "${JUDGE_TENSOR_PARALLEL_SIZE}" \
        --gpu_memory_utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
        --max_model_len "${JUDGE_MAX_MODEL_LEN}"
fi
