torchrun --nproc_per_node=8 --master_port=12345 tools/extract_bbox.py \
  --input_json annotated_results_gpt5.fixed.json \
  --output_dir data/parallel_outputs

cd /mnt/bn/xiangtai-training-data-video/scripts
bash run.sh