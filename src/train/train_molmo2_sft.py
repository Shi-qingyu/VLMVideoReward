from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train.argument import ModelArguments, SFTArguments
from src.train.sft_common import SFTModelSpec, TrainableModulePaths, train_sft


def _attn_implementation(attn_implementation: str) -> str:
    if attn_implementation == "flash_attention_2":
        return "sdpa"
    return attn_implementation


def load_model(
    model_args: ModelArguments,
    training_args: SFTArguments,
    attn_implementation: str,
) -> torch.nn.Module:
    torch_dtype = torch.bfloat16 if training_args.bf16 else None
    return AutoModelForImageTextToText.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        attn_implementation=_attn_implementation(attn_implementation),
        torch_dtype=torch_dtype,
    )


def load_processor(model_args: ModelArguments, training_args: SFTArguments):
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    if not hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = None
    return processor


MOLMO2_SPEC = SFTModelSpec(
    model_type="molmo2",
    default_attn_implementation="sdpa",
    load_model=load_model,
    load_processor=load_processor,
    trainable_paths=TrainableModulePaths(
        vision=(
            "vision_backbone",
            "vision_backbone.vit",
            "model.vision_backbone",
            "model.vision_backbone.vit",
            "model.model.vision_backbone",
            "model.model.vision_backbone.vit",
        ),
        projector=(
            "vision_backbone.image_pooling_2d",
            "vision_backbone.image_projector",
            "model.vision_backbone.image_pooling_2d",
            "model.vision_backbone.image_projector",
            "model.model.vision_backbone.image_pooling_2d",
            "model.model.vision_backbone.image_projector",
        ),
        language=(
            "language_model",
            "model.transformer",
            "model.model.transformer",
        ),
    ),
)


def train(attn_implementation: Optional[str] = None) -> None:
    train_sft(MOLMO2_SPEC, attn_implementation=attn_implementation)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
