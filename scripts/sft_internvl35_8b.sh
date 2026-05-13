#!/bin/bash

LLM_MODEL=${LLM_MODEL:-"OpenGVLab/InternVL3_5-8B-HF"} \
RUN_NAME=${RUN_NAME:-"internvl35-8b-baseline-bs${BATCH_SIZE:-1}-ga${GRAD_ACCUM_STEPS:-16}"} \
BATCH_SIZE=${BATCH_SIZE:-1} \
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-16} \
MASTER_PORT=${MASTER_PORT:-"12349"} \
bash scripts/sft_internvl35_4b.sh
