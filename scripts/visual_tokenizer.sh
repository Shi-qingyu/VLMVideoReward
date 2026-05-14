python tools/visualize_qwen3vl_visual_tokens.py \
  --model_path output/qwen3vl-4b-baseline-1e-bs4-ga4-merged-457 \
  --video data/videos/eval_0/1.mp4

python tools/visualize_internvl_visual_tokens.py \
  --model_path output/internvl35-4b-baseline-bs4-ga4 \
  --video data/videos/eval_0/1.mp4
  
python tools/visualize_molmo2_visual_tokens.py \
  --model_path output/molmo2-4b-baseline-bs4-ga4 \
  --video data/videos/eval_0/1.mp4 \
  --print_input_keys
