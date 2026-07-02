#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12353"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

deepspeed=${DEEPSPEED_CONFIG:-"${SCRIPT_DIR}/zero3.json"}
llm=${LLM_MODEL:-"OpenGVLab/InternVL3_5-4B-HF"}

lr=${LR:-"5e-5"}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}

entry_file=${ENTRY_FILE:-"${REPO_ROOT}/src/train/train_internvl_sft.py"}
datasets=${DATASETS:-"physics_violation_sampled_train"}

run_name=${RUN_NAME:-"internvl35-4b-physics-violation-bs${batch_size}-ga${grad_accum_steps}-fps${VIDEO_FPS:-2}-maxf${VIDEO_MAX_FRAMES:-20}-minf${VIDEO_MIN_FRAMES:-10}-imgsize${INTERNVL_IMAGE_SIZE:-448}-maxp${INTERNVL_MAX_PATCHES:-12}-minp${INTERNVL_MIN_PATCHES:-1}-lr${lr}"}
output_dir=${OUTPUT_DIR:-"${REPO_ROOT}/output/${run_name}"}

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --data_packing False \
    --tune_mm_vision ${TUNE_MM_VISION:-False} \
    --tune_mm_mlp ${TUNE_MM_MLP:-True} \
    --tune_mm_llm ${TUNE_MM_LLM:-True} \
    --using_cot True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS:-1} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --video_fps ${VIDEO_FPS:-2} \
    --video_max_frames ${VIDEO_MAX_FRAMES:-20} \
    --video_min_frames ${VIDEO_MIN_FRAMES:-10} \
    --internvl_image_size ${INTERNVL_IMAGE_SIZE:-448} \
    --internvl_max_patches ${INTERNVL_MAX_PATCHES:-12} \
    --internvl_min_patches ${INTERNVL_MIN_PATCHES:-1} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS:-100} \
    --save_total_limit ${SAVE_TOTAL_LIMIT:-1} \
    --learning_rate ${lr} \
    --weight_decay ${WEIGHT_DECAY:-0} \
    --warmup_ratio ${WARMUP_RATIO:-0.03} \
    --max_grad_norm ${MAX_GRAD_NORM:-1} \
    --lr_scheduler_type ${LR_SCHEDULER_TYPE:-cosine} \
    --logging_steps ${LOGGING_STEPS:-1} \
    --model_max_length ${MODEL_MAX_LENGTH:-8192} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-True} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --run_name ${run_name} \
    --report_to ${REPORT_TO:-wandb}
"

ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa} \
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
