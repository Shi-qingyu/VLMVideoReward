#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12345"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=output/qwen3vl-2b-baseline-1e-bs4-ga4-st  # Using HuggingFace model ID

# Reward func configuration
reward_func=acc_reward,format_reward,iou_reward
reward_func_weight=1.0,0.2,1.0

# Training hyperparameters
lr=2e-6
warmup_ratio=0.0
lr_scheduler_type=cosine
batch_size=4
grad_accum_steps=4
max_grad_norm=5
beta=0.04
weight_decay=0.01

# Training entry point
entry_file=qwenvl/train/train_qwen_grpo.py 

# Dataset configuration (replace with public dataset names)
datasets=videoreward_region

# Output configuration
run_name="qwen3vl-2b-baseline-1e-bs4-ga4-st-grpo"
output_dir=./output/${run_name}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
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
    --max_pixels 589824 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50 \
    --save_total_limit 5 \
    --learning_rate ${lr} \
    --warmup_ratio ${warmup_ratio} \
    --lr_scheduler_type ${lr_scheduler_type} \
    --reward_func ${reward_func} \
    --reward_func_weight ${reward_func_weight} \
    --beta ${beta} \
    --weight_decay 0 \
    --max_grad_norm ${max_grad_norm} \
    --logging_steps 1 \
    --max_new_tokens 512 \
    --max_input_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb
"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}

cd /mnt/bn/xiangtai-training-data-video/scripts
bash run.sh