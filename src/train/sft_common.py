from __future__ import annotations

import contextlib
import logging
import os
import pathlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch
import transformers
from packaging import version
from transformers import Trainer

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.dataset.data_processor import make_supervised_data_module
from src.train.argument import DataArguments, ModelArguments, SFTArguments

logger = logging.getLogger(__name__)

PROJECTOR_PARAMETER_KEYWORDS = (
    "merger",
    "embed_vision",
    "embed_audio",
    "multi_modal_projector",
    "mlp1",
    "resampler",
    "image_pooling_2d",
    "image_projector",
)
VISION_PARAMETER_KEYWORDS = (
    "visual",
    "vision_tower",
    "vision_model",
    "vision_backbone",
    "vpm",
)


@dataclass(frozen=True)
class TrainableModulePaths:
    vision: Sequence[str] = field(default_factory=tuple)
    projector: Sequence[str] = field(default_factory=tuple)
    language: Sequence[str] = field(default_factory=tuple)
    extra: Sequence[str] = ("lm_head",)


@dataclass(frozen=True)
class SFTModelSpec:
    model_type: str
    default_attn_implementation: str
    load_model: Callable[[ModelArguments, SFTArguments, str], torch.nn.Module]
    load_processor: Callable[[ModelArguments, SFTArguments], Any]
    trainable_paths: TrainableModulePaths
    prepare_processor: Optional[
        Callable[[Any, torch.nn.Module, DataArguments, SFTArguments], None]
    ] = None
    configure_data_args: Optional[
        Callable[[torch.nn.Module, DataArguments, SFTArguments], None]
    ] = None
    prepare_model_for_training: Optional[
        Callable[[torch.nn.Module, Any, DataArguments, SFTArguments], None]
    ] = None
    supports_flatten: bool = False
    lora_target_modules: Sequence[str] = ("q_proj", "k_proj", "v_proj", "o_proj")


def get_nested_attr(obj: Any, path: str) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def first_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def add_tokens_and_resize(tokenizer: Any, model: torch.nn.Module, tokens: Sequence[str]) -> None:
    num_added = tokenizer.add_tokens(list(tokens))
    if num_added > 0:
        model.resize_token_embeddings(len(tokenizer))


def prepare_tokenizer(processor: Any, training_args: SFTArguments) -> Any:
    tokenizer = processor.tokenizer
    tokenizer.model_max_length = training_args.model_max_length
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def _rng_safe_globals_context():
    if version.parse(torch.__version__).release < version.parse("2.6").release:
        return contextlib.nullcontext()

    np_core = np._core if version.parse(np.__version__) >= version.parse("2.0.0") else np.core
    allowlist = [
        np_core.multiarray._reconstruct,
        np.ndarray,
        np.dtype,
    ]
    try:
        if version.parse(np.__version__) < version.parse("1.25"):
            allowlist.append(type(np.dtype(np.uint32)))
        else:
            allowlist.append(np.dtypes.UInt32DType)
    except Exception:
        pass
    return torch.serialization.safe_globals(allowlist)


def patch_trainer_rng_state() -> None:
    if getattr(Trainer, "_vlm_reward_rng_state_patched", False):
        return

    old_load_rng_state = Trainer._load_rng_state

    def _patched_load_rng_state(self, checkpoint):
        with _rng_safe_globals_context():
            return old_load_rng_state(self, checkpoint)

    Trainer._load_rng_state = _patched_load_rng_state
    Trainer._vlm_reward_rng_state_patched = True


def _parameter_names_matching(model: torch.nn.Module, keywords: Sequence[str]) -> set[str]:
    return {
        name
        for name, _ in model.named_parameters()
        if any(keyword in name for keyword in keywords)
    }


def _parameter_group(
    parameters: list[torch.nn.Parameter],
    weight_decay: float,
    lr: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    if not parameters:
        return None
    group: dict[str, Any] = {"params": parameters, "weight_decay": weight_decay}
    if lr is not None:
        group["lr"] = lr
    return group


def create_sft_optimizer(self):
    opt_model = self.model
    if self.optimizer is not None:
        return self.optimizer

    decay_parameters = set(self.get_decay_parameter_names(opt_model))
    decay_parameters = {name for name in decay_parameters if "bias" not in name}
    projector_parameters = _parameter_names_matching(
        opt_model,
        PROJECTOR_PARAMETER_KEYWORDS,
    )
    vision_parameters = _parameter_names_matching(opt_model, VISION_PARAMETER_KEYWORDS)
    mm_projector_lr = getattr(self.args, "mm_projector_lr", None)
    vision_tower_lr = getattr(self.args, "vision_tower_lr", None)

    named_parameters = [
        (name, param)
        for name, param in opt_model.named_parameters()
        if param.requires_grad
    ]

    def collect(
        *,
        use_decay: bool,
        in_projector: Optional[bool] = None,
        in_vision: Optional[bool] = None,
    ) -> list[torch.nn.Parameter]:
        params = []
        for name, param in named_parameters:
            if (name in decay_parameters) != use_decay:
                continue
            if in_projector is not None and (name in projector_parameters) != in_projector:
                continue
            if in_vision is not None and (name in vision_parameters) != in_vision:
                continue
            params.append(param)
        return params

    groups: list[Optional[dict[str, Any]]]
    if mm_projector_lr:
        if vision_tower_lr:
            groups = [
                _parameter_group(
                    collect(use_decay=True, in_projector=False, in_vision=False),
                    self.args.weight_decay,
                ),
                _parameter_group(
                    collect(use_decay=False, in_projector=False, in_vision=False),
                    0.0,
                ),
                _parameter_group(
                    collect(use_decay=True, in_projector=False, in_vision=True),
                    self.args.weight_decay,
                    vision_tower_lr,
                ),
                _parameter_group(
                    collect(use_decay=False, in_projector=False, in_vision=True),
                    0.0,
                    vision_tower_lr,
                ),
                _parameter_group(
                    collect(use_decay=True, in_projector=True),
                    self.args.weight_decay,
                    mm_projector_lr,
                ),
                _parameter_group(
                    collect(use_decay=False, in_projector=True),
                    0.0,
                    mm_projector_lr,
                ),
            ]
        else:
            groups = [
                _parameter_group(
                    collect(use_decay=True, in_projector=False),
                    self.args.weight_decay,
                ),
                _parameter_group(collect(use_decay=False, in_projector=False), 0.0),
                _parameter_group(
                    collect(use_decay=True, in_projector=True),
                    self.args.weight_decay,
                    mm_projector_lr,
                ),
                _parameter_group(
                    collect(use_decay=False, in_projector=True),
                    0.0,
                    mm_projector_lr,
                ),
            ]
    else:
        groups = [
            _parameter_group(collect(use_decay=True), self.args.weight_decay),
            _parameter_group(collect(use_decay=False), 0.0),
        ]

    optimizer_grouped_parameters = [group for group in groups if group is not None]
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
    self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    return self.optimizer


def patch_trainer_optimizer() -> None:
    if getattr(Trainer, "_vlm_reward_sft_optimizer_patched", False):
        return
    Trainer.create_optimizer = create_sft_optimizer
    Trainer._vlm_reward_sft_optimizer_patched = True


def is_rank0_or_single_process() -> bool:
    return (
        not torch.distributed.is_available()
        or not torch.distributed.is_initialized()
        or torch.distributed.get_rank() == 0
    )


def _checkpoint_step(checkpoint_path: pathlib.Path) -> int:
    try:
        return int(checkpoint_path.name.rsplit("-", 1)[-1])
    except ValueError:
        return -1


def find_latest_complete_checkpoint(output_dir: str) -> Optional[str]:
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
        logger.warning(
            "Found checkpoint directories without trainer_state.json; treating them "
            "as incomplete and not resuming from them: %s",
            incomplete_checkpoints,
        )
    return None


def _set_module_trainable(module: Any, trainable: bool) -> None:
    if module is None:
        return
    for param in module.parameters():
        param.requires_grad = trainable


def set_trainable_modules(
    model: torch.nn.Module,
    training_args: SFTArguments,
    paths: TrainableModulePaths,
) -> None:
    for path in paths.vision:
        _set_module_trainable(get_nested_attr(model, path), training_args.tune_mm_vision)
    for path in paths.projector:
        _set_module_trainable(get_nested_attr(model, path), training_args.tune_mm_mlp)
    for path in paths.language:
        _set_module_trainable(get_nested_attr(model, path), training_args.tune_mm_llm)
    for path in paths.extra:
        _set_module_trainable(get_nested_attr(model, path), training_args.tune_mm_llm)


def _parameter_numel(param: torch.nn.Parameter) -> int:
    ds_numel = getattr(param, "ds_numel", None)
    if ds_numel is not None:
        return int(ds_numel)
    return int(param.numel())


def print_trainable_summary(model: torch.nn.Module) -> None:
    trainable = 0
    total = 0
    for param in model.parameters():
        count = _parameter_numel(param)
        total += count
        if param.requires_grad:
            trainable += count
    ratio = 100 * trainable / total if total else 0.0
    print(f"Trainable parameters: {trainable:,} / {total:,} ({ratio:.4f}%)")


def _enable_gradient_checkpointing_inputs(model: torch.nn.Module) -> None:
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
        return

    def make_inputs_require_grad(_module, _input, output):
        output.requires_grad_(True)

    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)


def _apply_lora(
    model: torch.nn.Module,
    training_args: SFTArguments,
    target_modules: Sequence[str],
) -> torch.nn.Module:
    from peft import LoraConfig, TaskType, get_peft_model

    print("LoRA enabled")
    lora_config = LoraConfig(
        r=training_args.lora_r or 64,
        lora_alpha=training_args.lora_alpha or 128,
        lora_dropout=training_args.lora_dropout or 0.05,
        target_modules=list(target_modules),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    return get_peft_model(model, lora_config)


def _set_use_cache(model: torch.nn.Module, value: Any) -> None:
    config = getattr(model, "config", None)
    if config is not None and hasattr(config, "use_cache"):
        config.use_cache = value


def _validate_training_mode(spec: SFTModelSpec, data_args: DataArguments, training_args: SFTArguments) -> None:
    del training_args
    if not spec.supports_flatten and (data_args.data_flatten or data_args.data_packing):
        raise ValueError(
            f"{spec.model_type} training currently uses standard padded batches. "
            "Set --data_flatten False and --data_packing False."
        )


def train_sft(
    spec: SFTModelSpec,
    attn_implementation: Optional[str] = None,
) -> None:
    patch_trainer_rng_state()
    patch_trainer_optimizer()

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, SFTArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    attn_implementation = attn_implementation or spec.default_attn_implementation

    data_args.model_type = spec.model_type
    setattr(training_args, "model_type", spec.model_type)
    _validate_training_mode(spec, data_args, training_args)
    os.makedirs(training_args.output_dir, exist_ok=True)

    model = spec.load_model(model_args, training_args, attn_implementation)
    if spec.configure_data_args is not None:
        spec.configure_data_args(model, data_args, training_args)

    print(
        f"initialized model {model_args.model_name_or_path} "
        f"as {model.__class__.__name__}"
    )
    processor = spec.load_processor(model_args, training_args)
    prepare_tokenizer(processor, training_args)
    if spec.prepare_processor is not None:
        spec.prepare_processor(processor, model, data_args, training_args)
    if spec.prepare_model_for_training is not None:
        spec.prepare_model_for_training(model, processor, data_args, training_args)

    previous_use_cache = getattr(getattr(model, "config", None), "use_cache", None)
    _set_use_cache(model, False)

    if training_args.gradient_checkpointing:
        _enable_gradient_checkpointing_inputs(model)

    if training_args.lora_enable:
        model = _apply_lora(model, training_args, spec.lora_target_modules)
    else:
        set_trainable_modules(model, training_args, spec.trainable_paths)
        if is_rank0_or_single_process():
            print_trainable_summary(model)

    data_module = make_supervised_data_module(processor, data_args=data_args)
    trainer = Trainer(
        model=model,
        processing_class=processor,
        args=training_args,
        **data_module,
    )

    resume_checkpoint = find_latest_complete_checkpoint(training_args.output_dir)
    if resume_checkpoint is not None:
        logger.info("checkpoint found, resume training from %s", resume_checkpoint)
        trainer.train(resume_from_checkpoint=resume_checkpoint)
    else:
        trainer.train()
    trainer.save_state()

    if previous_use_cache is not None:
        _set_use_cache(model, previous_use_cache)
    else:
        _set_use_cache(model, True)

    trainer.save_model(training_args.output_dir)
    processor.save_pretrained(training_args.output_dir)
