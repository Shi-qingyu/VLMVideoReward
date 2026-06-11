CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tools/qwen_judge.py \
  --input_file eval_results/internvl35-4b-bs4-ga4-t-merged-unique.json \
  --batch_size 64 \
  --gpu_memory_utilization 0.75