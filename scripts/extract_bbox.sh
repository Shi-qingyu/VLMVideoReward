torchrun --nproc_per_node=8 --master_port=12345 tools/extract_bbox.py \
  --input_json data/train_t_polished_v3_nouns_single.json \
  --output_dir data/parallel_outputs

cd /mnt/bn/xiangtai-training-data-video/scripts
bash run.sh