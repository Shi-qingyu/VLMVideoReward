#!/bin/bash

set -euo pipefail

cd "$(dirname "$0")/../.."

export DATA_MODE=video
export HYDRA_FULL_ERROR=1
export WANDB_MODE=${WANDB_MODE:-offline}

PRETRAINED_PATH=${PRETRAINED_PATH:-"../output/qwen3vl-2b-baseline-1e-bs4-ga4-region-457/checkpoint-350"}
TRAIN_FILES=${TRAIN_FILES:-"../data/train_region.json"}
VAL_FILES=${VAL_FILES:-"None"}
MEDIA_ROOT=${MEDIA_ROOT:-"../data"}
RUN_NAME=${RUN_NAME:-"qwenvl-grpo-verl"}
CKPT_SAVE_DIR=${CKPT_SAVE_DIR:-"./checkpoints/${RUN_NAME}"}
LOG_SAVE_DIR=${LOG_SAVE_DIR:-"./log/${RUN_NAME}/$(date +"%Y%m%d-%H%M%S")"}

mkdir -p "${CKPT_SAVE_DIR}" "${LOG_SAVE_DIR}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="[${TRAIN_FILES}]" \
    data.val_files="${VAL_FILES}" \
    data.train_batch_size=${TRAIN_BATCH_SIZE:-32} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH:-16384} \
    data.max_response_length=${MAX_RESPONSE_LENGTH:-1024} \
    data.video_key=videos \
    data.answer_key=ground_truth \
    data.return_raw_chat=True \
    data.media_root="${MEDIA_ROOT}" \
    data.data_source=qwenvl_video_judge \
    data.acc_reward_weight=${ACC_REWARD_WEIGHT:-1.0} \
    data.format_reward_weight=${FORMAT_REWARD_WEIGHT:-0.2} \
    data.iou_reward_weight=${IOU_REWARD_WEIGHT:-1.0} \
    reward_model.reward_manager=naive_multithreads \
    actor_rollout_ref.model.path="${PRETRAINED_PATH}" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-32} \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${PPO_MICRO_BATCH_SIZE_PER_GPU:-1} \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-0.2} \
    actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-0.3} \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${TENSOR_MODEL_PARALLEL_SIZE:-1} \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${ROLLOUT_LOGPROB_MB_PER_GPU:-1} \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB_PER_GPU:-1} \
    actor_rollout_ref.rollout.n=${NUM_GENERATIONS:-8} \
    actor_rollout_ref.rollout.temperature=${TEMPERATURE:-1.0} \
    actor_rollout_ref.rollout.top_p=${TOP_P:-0.95} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.75} \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_BATCHED_TOKENS:-24576} \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={video: 1}' \
    trainer.project_name=${PROJECT_NAME:-VLMVideoReward} \
    trainer.experiment_name="${RUN_NAME}" \
    trainer.logger="['console','wandb']" \
    trainer.nnodes=${NNODES:-1} \
    trainer.n_gpus_per_node=${NGPUS_PER_NODE:-8} \
    trainer.save_freq=${SAVE_FREQ:-20} \
    trainer.test_freq=${TEST_FREQ:--1} \
    trainer.total_epochs=${TOTAL_EPOCHS:-1} \
    trainer.default_local_dir="${CKPT_SAVE_DIR}" \
    trainer.val_before_train=False \
    trainer.critic_warmup=0 \
    trainer.num_workers=${NUM_WORKERS:-4} \
    2>&1 | tee "${LOG_SAVE_DIR}/train_log.txt"
