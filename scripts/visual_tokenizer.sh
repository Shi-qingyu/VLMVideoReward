#!/usr/bin/env bash
set -euo pipefail

VIDEO="${VIDEO:-data/videos/eval_0/1.mp4}"
OUT_DIR="${OUT_DIR:-output}"

python tools/visualize_qwen3vl_visual_tokens.py \
  --model_path output/qwen3vl-4b-baseline-1e-bs4-ga4-merged-457 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR"

python tools/visualize_internvl_visual_tokens.py \
  --model_path output/internvl35-4b-baseline-bs4-ga4 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR"
  
python tools/visualize_molmo2_visual_tokens.py \
  --model_path output/molmo2-4b-baseline-bs4-ga4 \
  --video "$VIDEO" \
  --output_dir "$OUT_DIR" \
  --print_input_keys
