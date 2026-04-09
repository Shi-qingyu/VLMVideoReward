export PATH="/mnt/shared-storage-user/zengxiangyu/miniconda3/bin:$PATH"
__conda_setup="$('/mnt/shared-storage-user/zengxiangyu/miniconda3/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/mnt/shared-storage-user/zengxiangyu/miniconda3/etc/profile.d/conda.sh" ]; then
        . "/mnt/shared-storage-user/zengxiangyu/miniconda3/etc/profile.d/conda.sh"
    else
        export PATH="/mnt/shared-storage-user/zengxiangyu/miniconda3/bin:$PATH"
    fi
fi
unset __conda_setup
conda deactivate
conda activate mini-o3
cd /your_local_path_to/Video-o3/RL
# ================================

FPS4_NEXTGQA_TEST=annodata/test/4fps_nextgqa_test.json
FPS4_CGBENCH_TEST=annodata/test/4fps_cgbench.json

CHARADES_TEST=annodata/test/charades_test.json

FPS4_VIDEOMME_TEST=annodata/test/4fps_videomme.json
FPS4_MLVU_TEST=annodata/test/4fps_mlvu_val.json
FPS4_VRBENCH_TEST=annodata/test/4fps_vrbench.json
FPS4_LVBENCH_TEST=annodata/test/4fps_lvbench.json
FPS4_LONGVIDEOBENCH_TEST=annodata/test/4fps_longvideobench.json
FPS4_VIDEOMMMU_TEST=annodata/test/4fps_videommmu.json
FPS4_MMVU_TEST=annodata/test/4fps_mmvu.json
FPS4_VIDEOHOLMES_TEST=annodata/test/4fps_videoholmes.json

TARGET_DATASET=$1
CKPT_NAME=$2
CKPT_STEP=$3

if [ -z "$TARGET_DATASET" ] || [ -z "$CKPT_NAME" ] || [ -z "$CKPT_STEP" ]; then
    echo "Usage: $0 DATASET CKPT_NAME CKPT_STEP"
    exit 1
fi

PRETRAINED_PATH="/root/s3/videogpu/videochat-o3/ckpt/${CKPT_NAME}/global_step_${CKPT_STEP}/actor/huggingface"

case $TARGET_DATASET in
    "NEXTGQA_TEST") TEST_FILE=$FPS4_NEXTGQA_TEST ;;
    "CGBENCH_TEST") TEST_FILE=$FPS4_CGBENCH_TEST ;;
    "CHARADES_TEST") TEST_FILE=$CHARADES_TEST ;;
    "VIDEOMME_TEST") TEST_FILE=$FPS4_VIDEOMME_TEST ;;
    "MLVU_TEST") TEST_FILE=$FPS4_MLVU_TEST ;;
    "VRBENCH_TEST") TEST_FILE=$FPS4_VRBENCH_TEST ;;
    "LVBENCH_TEST") TEST_FILE=$FPS4_LVBENCH_TEST ;;
    "LONGVIDEOBENCH_TEST") TEST_FILE=$FPS4_LONGVIDEOBENCH_TEST ;;
    "VIDEOMMMU_TEST") TEST_FILE=$FPS4_VIDEOMMMU_TEST ;;
    "MMVU_TEST") TEST_FILE=$FPS4_MMVU_TEST ;;
    "VIDEOHOLMES_TEST") TEST_FILE=$FPS4_VIDEOHOLMES_TEST ;;
    "LONGVIDEOREASON_TEST") TEST_FILE=$FPS4_LONGVIDEOREASON_TEST ;;
    *) echo "Unknown dataset: $TARGET_DATASET"; exit 1 ;;
esac

export VLLM_USE_V1=1
export WANDB_MODE=offline
export RUN_NAME="EVAL-${CKPT_NAME}-STEP${CKPT_STEP}-${TARGET_DATASET}"
export BASE_IMAGE_DIR="/root/s3"
export CKPT_SAVE_DIR="/root/s3/videogpu/videochat-o3/ckpt/${RUN_NAME}"
export LOG_SAVE_DIR="./log/A_eval/EVAL-${CKPT_NAME}/STEP_${CKPT_STEP}/${TARGET_DATASET}"
export WANDB_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/wandb_dir
export WANDB_ARTIFACT_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/artifacts_dir
export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
rm -rf ${LOG_SAVE_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${WANDB_ARTIFACT_DIR}
mkdir -p ${LOG_SAVE_DIR}

export DATA_MODE="video"

ray start --head --dashboard-host=0.0.0.0

python3 -m verl.trainer.main_ppo \
    trainer.val_only=True \
    algorithm.adv_estimator=grpo \
    hydra.run.dir=${LOG_SAVE_DIR}/hydra_outputs \
    data.system_prompt="tool_crop" \
    data.train_files=[${TEST_FILE}] \
    data.val_files=[${TEST_FILE}] \
    data.train_batch_size=32 \
    data.max_prompt_length=18432 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.video_key=video \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=1.0 \
    data.decay_penalty_weight=0 \
    data.general_qa_reward_fn="general_qa_tool" \
    data.gpt_extract_answer=True \
    data.extract_answer_tags="strict" \
    data.return_raw_chat=True \
    data.gpt_threads=16 \
    data.tool_call="crop" \
    data.use_tgt_size=False \
    data.max_pixels=16384 \
    data.min_pixels=512 \
    reward_model.reward_manager=naive_multithreads_tool \
    actor_rollout_ref.actor.ignore_exceed=True \
    actor_rollout_ref.model.path=${PRETRAINED_PATH} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.000 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.use_multi_turn_response_mask=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'video': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=12 \
    'actor_rollout_ref.rollout.val_limit_mm_per_prompt={'video': 12}' \
    actor_rollout_ref.rollout.use_raw_image=True \
    actor_rollout_ref.rollout.multi_turn_prompt_type="v2" \
    actor_rollout_ref.rollout.vllm_infer_batch_size=32 \
    actor_rollout_ref.rollout.mode="async" \
    actor_rollout_ref.actor.clip_ratio_high=0.3 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.rollout.use_relative_coordinates=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='Mini-o3' \
    trainer.experiment_name='Mini-o3-RL' \
    trainer.val_generations_to_log_to_wandb=512 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=25 \
    trainer.default_local_dir=${CKPT_SAVE_DIR} \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.log_training_rollouts_freq=5 \
    trainer.train_generations_to_log_to_wandb=256 \
    trainer.use_3drope=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    2>&1 | tee ${LOG_SAVE_DIR}/eval_log.txt