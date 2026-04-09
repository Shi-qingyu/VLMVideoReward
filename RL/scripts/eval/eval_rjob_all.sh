CKPT_NAME=GRPO-260122v1_suoha_clue3-NNODES4
CKPT_STEP=400




CKPT_PATH="/root/s3/videogpu/videochat-o3/ckpt/${CKPT_NAME}/global_step_${CKPT_STEP}/actor/huggingface"
if [ ! -f "${CKPT_PATH}/model-00004-of-00004.safetensors" ]; then
    echo "Error: ${CKPT_PATH}/model-00004-of-00004.safetensors does not exist!"
    exit 1
fi



datasets=(
    "MMVU_TEST"
    "VIDEOHOLMES_TEST"
    "LONGVIDEOBENCH_TEST"
    "VIDEOMME_TEST"
    "MLVU_TEST"
    "LVBENCH_TEST"
    "VIDEOMMMU_TEST"
)



for DATASET in "${datasets[@]}"; do
    echo "Submitting job for $DATASET..."
    rjob submit --name=EVAL-${DATASET}-${CKPT_NAME}-STEP${CKPT_STEP} \
        --gpu=8 --memory=1400000 --cpu=130 \
        --charged-group=intern9_gpu --private-machine=group --priority=6 \
        --mount=gpfs://gpfs1/zengxiangyu:/mnt/shared-storage-user/zengxiangyu \
        --custom-resources brainpp.cn/fuse=1 \
        --image=registry.h.pjlab.org.cn/ailab-intern9-intern9_gpu/zxy:videorl-20251118185838 \
        -- sh /your_local_path_to/Video-o3/RL/scripts/eval/eval_RL_video_single.sh $DATASET $CKPT_NAME $CKPT_STEP
done
