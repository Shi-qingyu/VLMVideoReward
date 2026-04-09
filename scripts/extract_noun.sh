CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python tools/extract_noun.py \
  --input_json data/train_fixed.json \
  --output_json data/train_nouns.json \
  --batch_size 64 \
  --gpu_memory_utilization 0.75 \
  --tensor_parallel_size 8