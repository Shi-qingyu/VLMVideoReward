CKPT_NAME=GRPO-260122v1_suoha_clue3-NNODES4
CKPT_STEP=400


ACTOR_PATH="/root/s3/videogpu/videochat-o3/ckpt/${CKPT_NAME}/global_step_${CKPT_STEP}/actor"

python3 tools/model_merger.py --local_dir ${ACTOR_PATH}

sleep 5
