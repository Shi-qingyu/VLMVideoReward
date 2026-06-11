MODEL_NAME=qwen3vl-4b-baseline-1e-bs4-ga4-t-merged-unique

python evaluation.py \
    --model_path output/${MODEL_NAME} \
    --dataset_use videoreward_eval_merged_unique

python tools/qwen_judge.py \
  --input_file eval_results/${MODEL_NAME}.json \
  --batch_size 64 \
  --gpu_memory_utilization 0.75