#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-"12346"}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=output/qwen3vl-2b-baseline/checkpoint-450  # Using HuggingFace model ID

# Training hyperparameters
lr=5e-5
batch_size=4
grad_accum_steps=4

# Training entry point
entry_file=qwenvl/train/train_qwen_grpo.py 

# Dataset configuration (replace with public dataset names)
datasets=videoreward

# Output configuration
run_name="qwen3vl-2b-baseline-grpo"
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
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --max_output_length 512 \
    --max_input_length 16384 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --run_name ${run_name} \
    --report_to wandb"

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}