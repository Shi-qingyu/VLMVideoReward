CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tools/qwen_judge.py \
  --input_file eval_results/qwen3vl-2b-vjepa21-distill-weight1.0-vision-lr5e-5-bs4-ga4-t-merged.json \
  --batch_size 64 \
  --gpu_memory_utilization 0.75