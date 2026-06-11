#!/usr/bin/env bash
set -euo pipefail

VIDEO="${VIDEO:-data/videos/eval_0/1.mp4}"
OUT_DIR="${OUT_DIR:-output}"

python tools/visualize_qwen3vl_visual_tokens.py \
  --model_path output/qwen3vl-2b-baseline-1e-bs4-ga4-merged-457/checkpoint-200 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR"

python tools/visualize_internvl_visual_tokens.py \
  --model_path output/internvl35-4b-baseline-bs4-ga4-fps2-maxf20-minf10-imgsize448-maxp12-minp1 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR"
  
python tools/visualize_molmo2_visual_tokens.py \
  --model_path output/molmo2-4b-baseline-bs4-ga4-fps2-maxf20-minf10-imgsize378-lr5e-5/checkpoint-600 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR"
