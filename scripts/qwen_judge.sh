CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tools/qwen_judge.py \
  --input_file eval_results/qwen3vl-2b-baseline-1e-bs4-ga4-st-grpo.json \
  --batch_size 32 \
  --gpu_memory_utilization 0.6