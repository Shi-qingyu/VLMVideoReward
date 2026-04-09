ray stop
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

VERTION_NAME="260122v1_suoha_768F_2FPS_fpt_clue3"

# export SELF_SET_OVERVIEW_FPS=1.0
# export SELF_SET_FPS_MAX_FRAMES=384
export RUN_NAME=GRPO-${VERTION_NAME}-NNODES${NNODES}
export DATA_MODE="video"
export VLLM_USE_V1=1
export WANDB_MODE=offline
export PRETRAINED_PATH="/mnt/shared-storage-user/zengxiangyu/Backup/Qwen2.5-VL-7B-Instruct"
# export PRETRAINED_PATH="/root/s3/videogpu/videochat-o3/ckpt/clue_stage2-multi3"
export BASE_IMAGE_DIR="/root/s3"
export CKPT_SAVE_DIR="/root/s3/videogpu/videochat-o3/ckpt/${RUN_NAME}"
export LOG_SAVE_DIR="./log/${RUN_NAME}/resume_$(date +"%Y%m%d-%H%M")"
export WANDB_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/wandb_dir
export WANDB_ARTIFACT_DIR=/mnt/shared-storage-user/zengxiangyu/tmp/cache/artifacts_dir
export TMPDIR=/tmp
export HYDRA_FULL_ERROR=1
mkdir -p ${CKPT_SAVE_DIR}
mkdir -p ${WANDB_DIR}
mkdir -p ${WANDB_ARTIFACT_DIR}
mkdir -p ${LOG_SAVE_DIR}

CHARADES=annodata/RL/charades_grounding_12408.json
CGBENCH_WT=annodata/RL/cgbench_correct_clue_single_w_tool_6764.json
LLaVid_M_WT=annodata/RL/llava-video_youtube_qa_mc_2_3_m_clue_multi_w_tool_13900.json
LLaVid_M_WOT=annodata/RL/llava-video_youtube_qa_mc_2_3_m_clue_multi_wo_tool_29523.json
LLaVid_S_WT=annodata/RL/llava-video_youtube_qa_mc_2_3_m_clue_single_w_tool_79848.json
LLaVid_S_WOT=annodata/RL/llava-video_youtube_qa_mc_2_3_m_clue_single_wo_tool_9946.json
LongVDB_WT=annodata/RL/longvideodb_gemini_clue_single_w_tool_7000.json
LongVideoReason_FREE=annodata/RL/longvideoreason_qa_from120to3600_freeform_9531.json
NEXTGQA_WT=annodata/RL/nextgqa_val_w_tool_2365.json
NEXTGQA_WOT=annodata/RL/nextgqa_val_wo_tool_702.json
SELFBUILT_1_WT=annodata/RL/selfbuilt_1_qa_f180to600_clue_single_w_tool_5796.json
SELFBUILT_2_WT=annodata/RL/selfbuilt_2_qa_f180to600_clue_single_w_tool_7491.json

SUBSET_CHARADES_TEST=annodata/test/subset/subset_charades_test_600.json
SUBSET_MLVU_TEST=annodata/test/subset/subset_4fps_mlvu_val_400.json
SUBSET_VIDEOMME_TEST=annodata/test/subset/subset_4fps_videomme_600.json

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    hydra.run.dir=${LOG_SAVE_DIR}/hydra_outputs \
    trainer.val_before_train=True \
    data.system_prompt="tool_crop" \
    data.train_files=[${CHARADES},${CGBENCH_WT},${LLaVid_M_WT},${LLaVid_M_WOT},${LLaVid_S_WT},${LLaVid_S_WOT},${LongVDB_WT},${LongVideoReason_FREE},${NEXTGQA_WT},${NEXTGQA_WOT},${SELFBUILT_1_WT},${SELFBUILT_2_WT}] \
    data.val_files=[${SUBSET_CHARADES_TEST},${SUBSET_MLVU_TEST},${SUBSET_VIDEOMME_TEST}] \
    data.train_batch_size=32 \
    data.max_prompt_length=18432 \
    data.max_response_length=8192 \
    data.image_key=images \
    data.video_key=video \
    data.answer_key=solution \
    data.mask_blank=False \
    data.acc_reward_weight=1.0 \
    data.format_reward_weight=1.0 \
    data.decay_penalty_weight=0.05 \
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
    actor_rollout_ref.rollout.max_num_batched_tokens=34816 \
    actor_rollout_ref.rollout.name=vllm_multi_turn_tool_call \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.max_generation_round=6 \
    'actor_rollout_ref.rollout.limit_mm_per_prompt={'video': 12}' \
    actor_rollout_ref.rollout.val_max_generation_round=6 \
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
    trainer.save_freq=5 \
    trainer.default_local_dir=${CKPT_SAVE_DIR} \
    trainer.test_freq=25 \
    trainer.total_epochs=1 \
    trainer.log_training_rollouts_freq=5 \
    trainer.train_generations_to_log_to_wandb=256 \
    trainer.use_3drope=True \
    trainer.rejection_sample=True \
    trainer.rejection_sample_multiplier=1 \
    2>&1 | tee ${LOG_SAVE_DIR}/train_log.txt