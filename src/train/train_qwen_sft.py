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

import torch
import transformers

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.train.trainer_sft import replace_qwen3_vl_attention_class
from src.train.trainer_distill import Qwen3VLDistillationTrainer

from transformers import AutoModel, AutoModelForImageTextToText, Qwen3VLForConditionalGeneration
from src.dataset.data_processor import make_supervised_data_module
from src.train.argument import (
    ModelArguments,
    DataArguments,
    SFTArguments,
)
from src.train.checkpoint_utils import (
    filter_state_dict_for_inference,
    strip_distill_only_weights_in_dir,
)
from transformers import AutoProcessor, Trainer

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


def _set_module_trainable(module, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def _set_modules_trainable(modules, trainable: bool) -> None:
    for module in modules:
        _set_module_trainable(module, trainable)


def _print_trainable_summary(model) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = param.numel()
        total += count
        if param.requires_grad:
            trainable += count
    ratio = 100 * trainable / total if total else 0
    print(
        f"Trainable parameters: {trainable:,} / {total:,} "
        f"({ratio:.4f}%)"
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
        vision_modules = [_get_nested_attr(model, "vision_model")]
        projector_modules = [_get_nested_attr(model, "mlp1")]
        language_modules = [_get_nested_attr(model, "language_model")]
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

    if model_type in {"internvl", "minicpmv"}:
        model_kwargs = {
            "cache_dir": training_args.cache_dir,
            "trust_remote_code": True,
            "torch_dtype": torch_dtype,
            "low_cpu_mem_usage": True,
        }
        if model_type == "internvl":
            model_kwargs["use_flash_attn"] = attn_implementation == "flash_attention_2"
        else:
            model_kwargs["attn_implementation"] = (
                "sdpa" if attn_implementation == "flash_attention_2" else attn_implementation
            )
        return AutoModel.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )

    raise ValueError(f"Unsupported model_type: {model_type}")


def _prepare_processor(processor, model, model_type: str) -> None:
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if model_type == "internvl" and hasattr(model, "img_context_token_id"):
        context_token = getattr(tokenizer, "context_image_token", "<IMG_CONTEXT>")
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(context_token)

    if model_type in {"gemma4", "internvl", "minicpmv"}:
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
    if model_type in {"gemma4", "internvl", "minicpmv"} and (
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
    if model_type in {"internvl", "minicpmv"} and training_args.distill_enable:
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

    print(f'the initlized model is {model_args.model_name_or_path} the class is {model.__class__.__name__}')
    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_type in {"internvl", "minicpmv"},
    )
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
            if hasattr(model.model, "print_trainable_parameters"):
                model.model.print_trainable_parameters()
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
    trainer_cls = Qwen3VLDistillationTrainer if training_args.distill_enable else Trainer
    trainer = trainer_cls(
        model=model, processing_class=processor, args=training_args, **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
