# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import sys
from pathlib import Path
from typing import Optional

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.train.trainer_sft import replace_qwen3_vl_attention_class
from src.train.trainer_distill import Qwen3VLDistillationTrainer

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    Qwen3VLForConditionalGeneration,
)
from src.dataset.data_processor import IGNORE_INDEX, make_supervised_data_module
from src.train.argument import (
    ModelArguments,
    DataArguments,
    SFTArguments,
)
from src.train.checkpoint_utils import (
    filter_state_dict_for_inference,
    strip_distill_only_weights_in_dir,
)
from transformers import Trainer

import contextlib
from packaging import version
import numpy as np

def _rng_safe_globals_context():
    # PyTorch < 2.6 doesn't need this
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core

    allowlist = [
        np_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ]

    # PyTorch docs note dtype classes may also need allowlisting depending on NumPy version
    try:
        if version.parse(np.__version__) < version.parse("1.25"):
            allowlist.append(type(np.dtype(np.uint32)))
        else:
            allowlist.append(np.dtypes.UInt32DType)
    except Exception:
        pass

    return torch.serialization.safe_globals(allowlist)


_old_load_rng_state = Trainer._load_rng_state

def _patched_load_rng_state(self, checkpoint):
    with _rng_safe_globals_context():
        return _old_load_rng_state(self, checkpoint)

Trainer._load_rng_state = _patched_load_rng_state

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def is_rank0_or_single_process() -> bool:
    return not torch.distributed.is_available() or not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _checkpoint_step(checkpoint_path: pathlib.Path) -> int:
    try:
        return int(checkpoint_path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def _find_latest_complete_checkpoint(output_dir: str) -> Optional[str]:
    checkpoints = sorted(
        pathlib.Path(output_dir).glob("checkpoint-*"),
        key=_checkpoint_step,
    )
    incomplete_checkpoints = []
    for checkpoint in reversed(checkpoints):
        if (checkpoint / "trainer_state.json").is_file():
            return str(checkpoint)
        incomplete_checkpoints.append(str(checkpoint))

    if incomplete_checkpoints:
        logging.warning(
            "Found checkpoint directories without trainer_state.json; treating them "
            "as incomplete and not resuming from them: %s",
            incomplete_checkpoints,
        )
    return None


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        strip_distill_only_weights_in_dir(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        inference_state_dict = filter_state_dict_for_inference(state_dict)
        cpu_state_dict = {key: value.cpu() for key, value in inference_state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _infer_model_type(model_name_or_path: str) -> str:
    model_name = model_name_or_path.lower()
    if "qwen3" in model_name and "vl" in model_name:
        return "qwen3vl"
    if "gemma-4" in model_name or "gemma4" in model_name:
        return "gemma4"
    if "internvl" in model_name:
        return "internvl"
    if "molmo2" in model_name:
        return "molmo2"
    if "minicpm-v" in model_name or "minicpmv" in model_name:
        return "minicpmv"
    raise ValueError(f"Unsupported model type: {model_name_or_path}")


def _get_nested_attr(obj, path: str):
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _is_hf_internvl_checkpoint(model_name_or_path: str) -> bool:
    return model_name_or_path.rstrip("/").lower().endswith("-hf")


def _token_to_id(tokenizer, token: str, required: bool = True) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        token_id = tokenizer.encode(token, add_special_tokens=False)
        token_id = token_id[0] if len(token_id) == 1 else None
    if token_id is None:
        if not required:
            return None
        raise ValueError(f"Could not resolve tokenizer id for InternVL token: {token}")
    return int(token_id)


def _patch_internvl_tokenizer(tokenizer):
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
        value = getattr(tokenizer, attr, None) or fallback
        setattr(tokenizer, attr, value)

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


def _load_processor(model_args, training_args, model_type: str):
    processor_kwargs = {
        "cache_dir": training_args.cache_dir,
        "trust_remote_code": model_type in {"internvl", "minicpmv", "molmo2"},
    }
    try:
        processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            **processor_kwargs,
        )
    except AttributeError as exc:
        if model_type != "internvl" or "start_image_token" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
        )
        tokenizer = _patch_internvl_tokenizer(tokenizer)
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

    if model_type == "internvl":
        _patch_internvl_tokenizer(processor.tokenizer)
    if model_type == "molmo2" and not hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = None
    return processor


def _set_module_trainable(module, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def _set_modules_trainable(modules, trainable: bool) -> None:
    for module in modules:
        _set_module_trainable(module, trainable)


def _parameter_numel(param: torch.nn.Parameter) -> int:
    ds_numel = getattr(param, "ds_numel", None)
    if ds_numel is not None:
        return int(ds_numel)
    return int(param.numel())


def _first_int(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def _print_trainable_summary(model) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = _parameter_numel(param)
        total += count
        if param.requires_grad:
            trainable += count
    ratio = 100 * trainable / total if total else 0
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({ratio:.4f}%)"
    )


def _truncate_for_dump(text: str, max_chars: int = 20000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return (
        text[:half]
        + f"\n\n... <truncated {omitted} chars> ...\n\n"
        + text[-half:]
    )


def _decode_token_ids(tokenizer, token_ids) -> str:
    if torch.is_tensor(token_ids):
        token_ids = token_ids.detach().cpu().tolist()
    return tokenizer.decode(
        [int(token_id) for token_id in token_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def _tensor_summary(value) -> str:
    if torch.is_tensor(value):
        return (
            f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype}, "
            f"device={value.device})"
        )
    if isinstance(value, list):
        preview = value[:2]
        return f"list(len={len(value)}, preview={preview})"
    if isinstance(value, tuple):
        preview = value[:2]
        return f"tuple(len={len(value)}, preview={preview})"
    if isinstance(value, dict):
        return f"dict(keys={list(value)})"
    return repr(value)


def _batch_index_value(value, index: int, batch_size: int):
    if isinstance(value, list) and len(value) == batch_size:
        return value[index]
    if isinstance(value, tuple) and len(value) == batch_size:
        return value[index]
    return value


def _has_complete_response(text: str) -> bool:
    return all(
        tag in text
        for tag in ("<think>", "</think>", "<answer>", "</answer>")
    )


def _has_chat_end_label(text: str) -> bool:
    return any(marker in text for marker in ("<|im_end|>", "<end_of_turn>", "</s>"))


def _dump_first_train_batch(trainer, tokenizer, training_args) -> None:
    if not bool(getattr(training_args, "dump_first_batch", True)):
        return
    if not trainer.is_world_process_zero():
        return

    dump_file = Path(getattr(training_args, "first_batch_dump_file", "first_train_batch.txt"))
    if not dump_file.is_absolute():
        dump_file = Path(training_args.output_dir) / dump_file
    dump_file.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "First training batch debug dump",
        f"output_dir: {training_args.output_dir}",
        f"model_max_length: {getattr(tokenizer, 'model_max_length', 'N/A')}",
        f"pad_token_id: {getattr(tokenizer, 'pad_token_id', None)}",
        f"eos_token_id: {getattr(tokenizer, 'eos_token_id', None)}",
        "",
    ]

    incomplete_examples = []
    try:
        train_dataloader = trainer.get_train_dataloader()
        batch = next(iter(train_dataloader))
        lines.append("Batch fields:")
        for key in sorted(batch):
            lines.append(f"  - {key}: {_tensor_summary(batch[key])}")
        lines.append("")

        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        attention_mask = batch.get("attention_mask")
        if not torch.is_tensor(input_ids) or not torch.is_tensor(labels):
            lines.append("input_ids/labels are missing or are not tensors; cannot decode batch.")
        else:
            batch_size = int(input_ids.shape[0]) if input_ids.dim() > 1 else 1
            input_ids = input_ids if input_ids.dim() > 1 else input_ids.unsqueeze(0)
            labels = labels if labels.dim() > 1 else labels.unsqueeze(0)
            if torch.is_tensor(attention_mask) and attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            for index in range(batch_size):
                sample_input_ids = input_ids[index].detach().cpu()
                sample_labels = labels[index].detach().cpu()
                if torch.is_tensor(attention_mask):
                    sample_mask = attention_mask[index].detach().cpu().bool()
                else:
                    pad_token_id = getattr(tokenizer, "pad_token_id", None)
                    sample_mask = (
                        sample_input_ids.ne(pad_token_id)
                        if pad_token_id is not None
                        else torch.ones_like(sample_input_ids, dtype=torch.bool)
                    )

                effective_input_ids = sample_input_ids[sample_mask]
                supervised_ids = sample_labels[sample_labels.ne(IGNORE_INDEX)]
                supervised_text = _decode_token_ids(tokenizer, supervised_ids)
                complete_response = _has_complete_response(supervised_text)
                has_chat_end_label = _has_chat_end_label(supervised_text)
                try:
                    model_max_length = int(getattr(tokenizer, "model_max_length"))
                except Exception:
                    model_max_length = 10**18
                if not complete_response:
                    incomplete_examples.append(index)

                lines.extend(
                    [
                        "=" * 100,
                        f"Example {index}",
                        f"input_token_count: {int(sample_mask.sum().item())}",
                        f"supervised_token_count: {int(supervised_ids.numel())}",
                        (
                            "hit_model_max_length_or_truncated: "
                            f"{int(sample_mask.sum().item()) >= model_max_length}"
                        ),
                        f"complete_think_answer_labels: {complete_response}",
                        f"has_chat_end_label: {has_chat_end_label}",
                        f"distill_image_paths: {_batch_index_value(batch.get('distill_image_paths'), index, batch_size)}",
                        f"distill_video_paths: {_batch_index_value(batch.get('distill_video_paths'), index, batch_size)}",
                        "",
                        "[SUPERVISED LABEL TEXT: labels != -100]",
                        _truncate_for_dump(supervised_text),
                        "",
                        "[FULL INPUT TEXT]",
                        _truncate_for_dump(_decode_token_ids(tokenizer, effective_input_ids)),
                        "",
                    ]
                )

    except Exception as exc:
        logging.exception("Failed to dump first training batch.")
        lines.append(f"ERROR while dumping first training batch: {type(exc).__name__}: {exc}")

    dump_file.write_text("\n".join(lines), encoding="utf-8")
    logging.info("Saved first training batch debug dump to %s", dump_file)
    if incomplete_examples:
        logging.warning(
            "First training batch has examples without complete <think>/<answer> labels: %s. "
            "Check %s",
            incomplete_examples,
            dump_file,
        )


def set_model(training_args, model, model_type: str):
    if model_type == "qwen3vl":
        vision_modules = [_get_nested_attr(model, "model.visual")]
        projector_modules = [_get_nested_attr(model, "model.visual.merger")]
        language_modules = [_get_nested_attr(model, "model.language_model")]
    elif model_type == "gemma4":
        vision_modules = [
            _get_nested_attr(model, "model.vision_tower"),
        ]
        projector_modules = [
            _get_nested_attr(model, "model.embed_vision"),
            _get_nested_attr(model, "model.embed_audio"),
            _get_nested_attr(model, "model.multi_modal_projector"),
        ]
        language_modules = [_get_nested_attr(model, "model.language_model")]
    elif model_type == "internvl":
        vision_modules = [
            _get_nested_attr(model, "vision_model"),
            _get_nested_attr(model, "model.vision_tower"),
        ]
        projector_modules = [
            _get_nested_attr(model, "mlp1"),
            _get_nested_attr(model, "model.multi_modal_projector"),
        ]
        language_modules = [
            _get_nested_attr(model, "language_model"),
            _get_nested_attr(model, "model.language_model"),
        ]
    elif model_type == "molmo2":
        vision_modules = [
            _get_nested_attr(model, "vision_backbone"),
            _get_nested_attr(model, "vision_backbone.vit"),
            _get_nested_attr(model, "model.vision_backbone"),
            _get_nested_attr(model, "model.vision_backbone.vit"),
            _get_nested_attr(model, "model.model.vision_backbone"),
            _get_nested_attr(model, "model.model.vision_backbone.vit"),
        ]
        projector_modules = [
            _get_nested_attr(model, "vision_backbone.image_pooling_2d"),
            _get_nested_attr(model, "vision_backbone.image_projector"),
            _get_nested_attr(model, "model.vision_backbone.image_pooling_2d"),
            _get_nested_attr(model, "model.vision_backbone.image_projector"),
            _get_nested_attr(model, "model.model.vision_backbone.image_pooling_2d"),
            _get_nested_attr(model, "model.model.vision_backbone.image_projector"),
        ]
        language_modules = [
            _get_nested_attr(model, "language_model"),
            _get_nested_attr(model, "model.transformer"),
            _get_nested_attr(model, "model.model.transformer"),
        ]
    elif model_type == "minicpmv":
        vision_modules = [_get_nested_attr(model, "vpm")]
        projector_modules = [_get_nested_attr(model, "resampler")]
        language_modules = [_get_nested_attr(model, "llm")]
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    _set_modules_trainable(vision_modules, training_args.tune_mm_vision)
    _set_modules_trainable(projector_modules, training_args.tune_mm_mlp)
    _set_modules_trainable(language_modules, training_args.tune_mm_llm)
    _set_module_trainable(getattr(model, "lm_head", None), training_args.tune_mm_llm)


def _load_model(model_args, training_args, model_type: str, attn_implementation: str):
    torch_dtype = torch.bfloat16 if training_args.bf16 else None
    if model_type == "qwen3vl":
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=torch_dtype,
        )

    if model_type == "gemma4":
        gemma_attn_impl = (
            "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
        )
        return AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=gemma_attn_impl,
            torch_dtype=torch_dtype,
        )

    if model_type == "internvl":
        if _is_hf_internvl_checkpoint(model_args.model_name_or_path):
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

    if model_type == "minicpmv":
        model_kwargs = {
            "cache_dir": training_args.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        model_kwargs["attn_implementation"] = (
            "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
        )
        return AutoModel.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    if model_type == "molmo2":
        molmo_attn_impl = (
            "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
        )
        return AutoModelForImageTextToText.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            trust_remote_code=True,
            attn_implementation=molmo_attn_impl,
            torch_dtype=torch_dtype,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _prepare_processor(processor, model, model_type: str) -> None:
    tokenizer = processor.tokenizer
    if model_type == "internvl":
        _patch_internvl_tokenizer(tokenizer)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_type == "internvl" and hasattr(model, "img_context_token_id"):
        context_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(context_token)

    if model_type in {"gemma4", "internvl", "minicpmv", "molmo2"}:
        return

    num_added = tokenizer.add_tokens(
        ["<think>", "</think>", "<answer>", "</answer>", "<box>", "</box>", "<t>", "</t>"]
    )
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SFTArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    setattr(
        data_args,
        "distill_share_student_video_sampling",
        bool(getattr(training_args, "distill_enable", False)),
    )

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)

    model_type = _infer_model_type(model_args.model_name_or_path)
    data_args.model_type = model_type
    setattr(training_args, "model_type", model_type)
    if model_type in {"gemma4", "internvl", "minicpmv", "molmo2"} and (
        data_args.data_flatten or data_args.data_packing
    ):
        raise ValueError(
            f"{model_type} training currently uses standard padded batches. "
            "Set --data_flatten False and --data_packing False."
        )
    if model_type == "gemma4" and training_args.distill_enable:
        raise ValueError(
            "Gemma4 SFT is supported, but V-JEPA visual distillation is still "
            "wired to Qwen3VL visual token geometry. Set --distill_enable False."
        )
    if model_type in {"internvl", "minicpmv", "molmo2"} and training_args.distill_enable:
        raise ValueError(
            f"{model_type} SFT is supported, but V-JEPA visual distillation is "
            "currently wired to Qwen3VL visual token geometry. Set --distill_enable False."
        )

    model = _load_model(
        model_args=model_args,
        training_args=training_args,
        model_type=model_type,
        attn_implementation=attn_implementation,
    )
    if model_type == "internvl":
        vision_config = getattr(model.config, "vision_config", None)
        setattr(
            data_args,
            "internvl_model_image_size",
            _first_int(getattr(vision_config, "image_size", None)),
        )
        setattr(
            data_args,
            "internvl_patch_size",
            _first_int(getattr(vision_config, "patch_size", None)),
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
            model_type == "internvl"
            and (
                hasattr(model, "img_context_token_id")
                or _get_nested_attr(model, "vision_model") is not None
            )
        ),
    )

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = _load_processor(model_args, training_args, model_type)
    processor.tokenizer.model_max_length = training_args.model_max_length
    _prepare_processor(processor, model, model_type)

    if model_type == "qwen3vl" and (data_args.data_flatten or data_args.data_packing):
        replace_qwen3_vl_attention_class()
    model.config.use_cache = False

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        lora_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Qwen 的 attention 线性层
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_config)
    else:
        set_model(training_args, model, model_type=model_type)

        if is_rank0_or_single_process():
            inner_model = getattr(model, "model", model)
            if hasattr(inner_model, "print_trainable_parameters"):
                inner_model.print_trainable_parameters()
            _print_trainable_summary(model)

    if training_args.distill_enable and not training_args.tune_mm_vision:
        logging.warning(
            "Visual distillation is enabled while tune_mm_vision=False. "
            "This will train the distillation projector, but the vision tower itself stays frozen."
        )
    if training_args.distill_enable and training_args.tune_mm_llm:
        logging.warning(
            "Visual distillation is enabled while tune_mm_llm=True. "
            "This often destabilizes generation because the language model moves together with the vision tower. "
            "Prefer freezing the LLM and only tuning the vision tower plus merger/projectors first."
        )
    
    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Qwen3VLDistillationTrainer(
        model=model, processing_class=processor, args=training_args, **data_module
    )
    _dump_first_train_batch(trainer, processor.tokenizer, training_args)

    resume_checkpoint = _find_latest_complete_checkpoint(training_args.output_dir)
    if resume_checkpoint is not None:
        logging.info("checkpoint found, resume training from %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
