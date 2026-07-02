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
        attn_implementation=_attn_implementation(attn_implementation),
        torch_dtype=torch_dtype,
    )


def load_processor(model_args: ModelArguments, training_args: SFTArguments):
    return AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )


GEMMA_SPEC = SFTModelSpec(
    model_type="gemma4",
    default_attn_implementation="sdpa",
    load_model=load_model,
    load_processor=load_processor,
    trainable_paths=TrainableModulePaths(
        vision=("model.vision_tower",),
        projector=(
            "model.embed_vision",
            "model.embed_audio",
            "model.multi_modal_projector",
        ),
        language=("model.language_model",),
    ),
)


def train(attn_implementation: Optional[str] = None) -> None:
    train_sft(GEMMA_SPEC, attn_implementation=attn_implementation)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "sdpa"))
