#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12351"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# DeepSpeed configuration
deepspeed=${DEEPSPEED_CONFIG:-"${SCRIPT_DIR}/zero3.json"}

# Model configuration
llm=${LLM_MODEL:-"Qwen/Qwen3-VL-2B-Instruct"}

# Training hyperparameters
lr=${LR:-5e-5}
batch_size=${BATCH_SIZE:-4}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
tune_mm_llm=${TUNE_MM_LLM:-True}
tune_mm_mlp=${TUNE_MM_MLP:-True}
tune_mm_vision=${TUNE_MM_VISION:-False}

# Training entry point
entry_file=${ENTRY_FILE:-"${REPO_ROOT}/src/train/train_qwen_sft.py"}

# Dataset configuration
datasets=${DATASETS:-"human_distortion_sampled_train"}

# Output configuration
run_name=${RUN_NAME:-"qwen3vl-2b-human-distortion-bs${batch_size}-ga${grad_accum_steps}"}
output_dir=${OUTPUT_DIR:-"${REPO_ROOT}/output/${run_name}"}

args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten True \
    --tune_mm_vision ${tune_mm_vision} \
    --tune_mm_mlp ${tune_mm_mlp} \
    --tune_mm_llm ${tune_mm_llm} \
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
    --video_max_pixels ${VIDEO_MAX_PIXELS:-4194304} \
    --video_min_pixels ${VIDEO_MIN_PIXELS:-1048576} \
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
    --model_max_length ${MODEL_MAX_LENGTH:-16384} \
    --gradient_checkpointing ${GRADIENT_CHECKPOINTING:-True} \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --run_name ${run_name} \
    --report_to ${REPORT_TO:-wandb}
"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --nnodes=${NNODES} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
