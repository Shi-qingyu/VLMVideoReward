#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12347"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# DeepSpeed configuration
deepspeed=${DEEPSPEED_CONFIG:-"./scripts/zero3.json"}

# Model configuration
llm=${LLM_MODEL:-"google/gemma-4-E4B-it"}

# Training hyperparameters
lr=${LR:-"5e-5"}
batch_size=${BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}

# Training entry point
entry_file=${ENTRY_FILE:-"src/train/train_gemma_sft.py"}

# Dataset configuration
datasets=${DATASETS:-"videoreward_t_merged"}

# Output configuration
run_name=${RUN_NAME:-"gemma4-e4b-baseline-bs${batch_size}-ga${grad_accum_steps}"}
output_dir=${OUTPUT_DIR:-"./output/${run_name}"}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --data_flatten False \
    --data_packing False \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --using_cot True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 1 \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size $((batch_size*2)) \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --gemma4_max_soft_tokens ${GEMMA4_MAX_SOFT_TOKENS:-280} \
    --video_fps ${VIDEO_FPS:-2} \
    --video_max_frames ${VIDEO_MAX_FRAMES:-20} \
    --video_min_frames ${VIDEO_MIN_FRAMES:-10} \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps ${SAVE_STEPS:-500} \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length ${MODEL_MAX_LENGTH:-8192} \
    --gradient_checkpointing True \
    --dataloader_num_workers ${DATALOADER_NUM_WORKERS:-8} \
    --run_name ${run_name} \
    --report_to ${REPORT_TO:-wandb}
"

torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
