# QwenVL Training Framework

This repository provides a training framework for Qwen VL models. The are two steps to use our repo:

1. Customize your dataset: downloading data, implement the config
2. Modify training scripts:

## Repository Structure

The `src` directory contains the following components:

### `train/`
- `sft_common.py`: Shared SFT runner, optimizer patch, checkpoint resume/save, and LoRA utilities
- `train_qwen_sft.py`: Qwen3-VL SFT entry and Qwen-specific packed-batch attention patch
- `train_gemma_sft.py`: Gemma SFT entry
- `train_internvl_sft.py`: InternVL SFT entry
- `train_minicpmv_sft.py`: MiniCPM-V SFT entry
- `train_molmo2_sft.py`: Molmo2 SFT entry
- `argument.py`: Dataclasses for model, data and training arguments

### `evaluation/`
- `evaluation_common.py`: Shared evaluation loop, metrics, HF/vLLM generation helpers
- `evaluation_qwen.py`: Qwen-VL evaluation entry
- `evaluation_internvl.py`: InternVL evaluation entry
- `evaluation_gemma4.py`: Gemma evaluation entry
- `evaluation_minicpmv.py`: MiniCPM-V evaluation entry
- `evaluation_molmo2.py`: Molmo2 evaluation entry

### `dataset/`
- `__init__.py`: Contains datasets configs
- `data_processor.py`: Data processing module for QwenVL models
- `rope2d.py`: Provide RoPE implementation

## Requirements

You could install packages through:

```bash
pip install -r requirements.txt
```

## Custom Dataset Configuration

The customized data should have the format like:

### Example Instances:

1. **Single Image Example**:
```json
{
    "images": "images/001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat's the main object in this picture?"
        },
        {
            "from": "gpt",
            "value": "A red apple on a wooden table"
        }
    ]
}
```

2. **Multi-Image Example**:
```json
{
    "images": ["cats/001.jpg", "cats/002.jpg"],
    "conversations": [
        {
            "from": "human",
            "value": "<image>\n<image>\nWhat are the differences between these two cats?"
        },
        {
            "from": "gpt",
            "value": "The first cat is an orange tabby with short fur and green eyes, while the second is a gray Siamese with blue eyes and pointed coloration. They also appear to be in different environments - the first is indoors on a couch, the second is outdoors in a garden."
        }
    ]
}
```

3. **Video Example**:
```json
{
    "videos": "videos/005.mp4",
    "conversations": [
        {
            "from": "human",
            "value": "<video>\nWhat caused the blue object to move?\nOptions:\n(A) Gravity\n(B) Collision\n(C) Magnetic force"
        },
        {
            "from": "gpt",
            "value": "Answer: (B) Collision"
        }
    ]
}
```

4. **Grounding Example**:
```json
{
    "images": "demo/COCO_train2014_000000580957.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nLocate house in this image and output the bbox coordinates in JSON format."
        },
        {
            "from": "gpt",
            "value": "{\n"bbox_2d": [135, 114, 1016, 672]\n}"
        }
    ]
}
```

5. **Packed Data Example**:
```json
[
    {
        "images": "images/001.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A red apple on a wooden table"
            }
        ]
    },
    {
        "images": "images/002.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "<image>\nWhat's the main object in this picture?"
            },
            {
                "from": "gpt",
                "value": "A green orange on a plastic table"
            }
        ]
    }
]
```

The examples above illustrate the expected JSON structure and can be adapted directly for training.

### Dataset config for training

To add or modify datasets for training, follow these steps:

### Dataset Definition Structure

1. **Create a dataset dictionary** in the format in the file `src/dataset/__init__.py`:
```python
DATASET_NAME = {
    "annotation_path": "/path/to/annotations.json",
    "data_path": "/path/to/image/data",  # Can be empty if paths are in annotations
}
```

2. **Register your dataset** by adding it to the `data_dict`:
```python
data_dict = {
    "your_dataset_name": DATASET_NAME,
    # ... other datasets
}
```

### Sampling Rate Control

You can optionally specify sampling rates by appending `%X` to the dataset name:
- `"dataset_name%50"` will sample 50% of the data
- `"dataset_name%20"` will sample 20% of the data

### Notes  
- The `annotation_path` should point to a JSON or JSONL file containing your dataset annotations.  
- The `data_path` can be left empty if the image paths in the annotations are absolute.  
- Sampling rates are applied per-dataset when multiple datasets are specified.  
- Some datasets you can use directly: `nyu-visionx/Cambrian-10M`, `lmms-lab/LLaVA-NeXT-Data`, `FreedomIntelligence/ALLaVA-4V`, `TIGER-Lab/VisualWebInstruct`.  
- The training data should strictly follow this format:  
  - One `<image>` tag in the question must correspond to exactly one image file  
  - Similarly, `<video>` tags must correspond to video files  
  - These special tokens should not appear in the answer text  
- For open source data that might have missing images or other issues, you can verify data completeness using `tools/check_image.py`.  


## Usage

### SFT
To train the model, make sure to update the `dataset` field in the training script `scripts/sft_2b.sh`:

```bash
bash scripts/sft_2b.sh
```

The script accepts arguments in three categories:

   - Flags to control which components to tune (`tune_mm_vision`, `tune_mm_mlp`, `tune_mm_llm`). If trained with both image and video data, tune_mm_vision should be False: `tune_mm_vision=False`
   - `data_flatten` flag means data in a batch are concat into one sequence
   - `data_packing` requires preprocess with `tools/pack_data.py`
   - Training hyperparameters, the suggested learning rate is from 1e-6 to 2e-7
   - Training resolution is critical for the model performances, hence `--max_pixels` and `--min_pixels` should be properly set
   - Training with a Qwen2.5-VL-32B model typically requires 8x80G GPUs
   - `"_attn_implementation": "flash_attention_2",` could be add in the config.json of the model to use flash attention.
   - The Qwen3VL MoE model does not support DeepSpeed with ZeRO-3. Additionally, Hugging Face’s official implementation does not include support for load balancing loss currently.
