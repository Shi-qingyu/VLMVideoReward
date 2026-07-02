#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH=/path/to/checkpoint or MODEL_PATH=hf/model-id}

EVAL_ENTRY=${EVAL_ENTRY:-"${REPO_ROOT}/src/evaluation/evaluation_qwen.py"}
MODEL_TYPE=${MODEL_TYPE:-auto}
BACKEND=${BACKEND:-vllm}
OUTPUT_ROOT=${OUTPUT_ROOT:-"${REPO_ROOT}/eval_results/single_issue"}
EVAL_DATASETS=${EVAL_DATASETS:-"human_distortion_sampled_test,physics_violation_sampled_test,product_consistency_sampled_test"}

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

IFS=',' read -r -a datasets <<< "${EVAL_DATASETS}"
mkdir -p "${OUTPUT_ROOT}"

for dataset in "${datasets[@]}"; do
    dataset=$(echo "${dataset}" | xargs)
    if [[ -z "${dataset}" ]]; then
        continue
    fi

    dataset_output_dir="${OUTPUT_ROOT}/${dataset}"
    mkdir -p "${dataset_output_dir}"

    echo "==== Evaluating ${dataset} ===="
    python "${EVAL_ENTRY}" \
        --model_path "${MODEL_PATH}" \
        --model_type "${MODEL_TYPE}" \
        --backend "${BACKEND}" \
        --dataset_use "${dataset}" \
        --output_dir "${dataset_output_dir}" \
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

    result_json=$(find "${dataset_output_dir}" -maxdepth 1 -type f -name '*.json' \
        ! -name '*_metrics.json' \
        ! -name '*_judgment.json' \
        ! -name '*_reasoning_consistency.json' \
        | sort | head -n 1)

    if [[ -z "${result_json}" ]]; then
        echo "No evaluation result JSON found in ${dataset_output_dir}" >&2
        exit 1
    fi

    if [[ "${RUN_JUDGE}" == "true" ]]; then
        judgment_json="${result_json%.json}_reasoning_consistency.json"
        echo "==== Judging reasoning consistency for ${dataset} ===="
        python tools/qwen_judge.py \
            --model_name_or_path "${JUDGE_MODEL}" \
            --input_file "${result_json}" \
            --output_file "${judgment_json}" \
            --batch_size "${JUDGE_BATCH_SIZE}" \
            --max_new_tokens "${JUDGE_MAX_NEW_TOKENS}" \
            --tensor_parallel_size "${JUDGE_TENSOR_PARALLEL_SIZE}" \
            --gpu_memory_utilization "${JUDGE_GPU_MEMORY_UTILIZATION}" \
            --max_model_len "${JUDGE_MAX_MODEL_LEN}"
    fi
done
