from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train.argument import DataArguments, ModelArguments, SFTArguments
from src.train.sft_common import (
    SFTModelSpec,
    TrainableModulePaths,
    first_int,
    get_nested_attr,
    train_sft,
)


def _is_hf_checkpoint(model_name_or_path: str) -> bool:
    return model_name_or_path.rstrip("/").lower().endswith("-hf")


def _token_to_id(tokenizer, token: str, required: bool = True) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        token_ids = tokenizer.encode(token, add_special_tokens=False)
        token_id = token_ids[0] if len(token_ids) == 1 else None
    if token_id is None:
        if not required:
            return None
        raise ValueError(f"Could not resolve tokenizer id for InternVL token: {token}")
    return int(token_id)


def patch_tokenizer(tokenizer):
    required_tokens = {
        "start_image_token": "<img>",
        "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>",
    }
    optional_tokens = {
        "image_token": "<image>",
        "video_token": "<video>",
    }
    for attr, fallback in {**required_tokens, **optional_tokens}.items():
        setattr(tokenizer, attr, getattr(tokenizer, attr, None) or fallback)

    tokenizer.start_image_token_id = _token_to_id(tokenizer, tokenizer.start_image_token)
    tokenizer.end_image_token_id = _token_to_id(tokenizer, tokenizer.end_image_token)
    tokenizer.context_image_token_id = _token_to_id(
        tokenizer,
        tokenizer.context_image_token,
    )
    for attr in optional_tokens:
        token_id = _token_to_id(tokenizer, getattr(tokenizer, attr), required=False)
        if token_id is not None:
            setattr(tokenizer, f"{attr}_id", token_id)
    return tokenizer


def load_model(
    model_args: ModelArguments,
    training_args: SFTArguments,
    attn_implementation: str,
) -> torch.nn.Module:
    torch_dtype = torch.bfloat16 if training_args.bf16 else None
    if _is_hf_checkpoint(model_args.model_name_or_path):
        return AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

    return AutoModel.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_flash_attn=attn_implementation == "flash_attention_2",
    )


def load_processor(model_args: ModelArguments, training_args: SFTArguments):
    processor_kwargs = {
        "cache_dir": training_args.cache_dir,
        "trust_remote_code": True,
    }
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            **processor_kwargs,
        )
    except AttributeError as exc:
        if "start_image_token" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        tokenizer = patch_tokenizer(tokenizer)
        try:
            processor = AutoProcessor.from_pretrained(
                model_args.model_name_or_path,
                tokenizer=tokenizer,
                **processor_kwargs,
            )
        except AttributeError as second_exc:
            if "start_image_token" not in str(second_exc):
                raise
            from transformers import AutoVideoProcessor
            from transformers.models.internvl.processing_internvl import InternVLProcessor

            config = AutoConfig.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
            image_processor = AutoImageProcessor.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
            video_processor = AutoVideoProcessor.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                trust_remote_code=True,
            )
            processor = InternVLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                image_seq_length=int(getattr(config, "image_seq_length", 256)),
            )

    patch_tokenizer(processor.tokenizer)
    return processor


def configure_data_args(
    model: torch.nn.Module,
    data_args: DataArguments,
    training_args: SFTArguments,
) -> None:
    del training_args
    vision_config = getattr(model.config, "vision_config", None)
    setattr(
        data_args,
        "internvl_model_image_size",
        first_int(getattr(vision_config, "image_size", None)),
    )
    setattr(
        data_args,
        "internvl_patch_size",
        first_int(getattr(vision_config, "patch_size", None)),
    )
    setattr(
        data_args,
        "internvl_downsample_ratio",
        float(getattr(model.config, "downsample_ratio", 0.5)),
    )
    setattr(
        data_args,
        "internvl_use_image_flags",
        bool(
            hasattr(model, "img_context_token_id")
            or get_nested_attr(model, "vision_model") is not None
        ),
    )


def prepare_processor(processor, model, data_args, training_args) -> None:
    del data_args, training_args
    tokenizer = patch_tokenizer(processor.tokenizer)
    if hasattr(model, "img_context_token_id"):
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.context_image_token
        )


INTERNVL_SPEC = SFTModelSpec(
    model_type="internvl",
    default_attn_implementation="flash_attention_2",
    load_model=load_model,
    load_processor=load_processor,
    configure_data_args=configure_data_args,
    prepare_processor=prepare_processor,
    trainable_paths=TrainableModulePaths(
        vision=(
            "vision_model",
            "model.vision_tower",
        ),
        projector=(
            "mlp1",
            "model.multi_modal_projector",
        ),
        language=(
            "language_model",
            "model.language_model",
        ),
    ),
)


def train(attn_implementation: Optional[str] = None) -> None:
    train_sft(INTERNVL_SPEC, attn_implementation=attn_implementation)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
