import functools
import os
from pathlib import Path
from typing import Any

import torch
from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
    StoppingCriteria,
    StoppingCriteriaList,
)


DEFAULT_VIDEO_PATH = "data/videos/eval_0/1.mp4"
DEFAULT_VIDEO_FPS = float(os.environ.get("VIDEO_FPS", "2"))
DEFAULT_PROMPT = (
    "A young Black man with a beard walks through an aisle of a brightly lit toy store, surrounded by colorful shelves. "
    "He pauses in front of a shelf displaying puzzle sets, picks up a puzzle set in both hands, examines the pieces closely, "
    "and smiles at the memories of his own childhood. The camera remains steady, capturing his actions and the vibrant store setting."
)
QUESTION_TEMPLATE = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\n"
    "Evaluate the video according to the following dimensions.\n"
    "Video Quality: whether the video is free from major visual defects, including blur, lack of detail, "
    "poor texture, lighting issues, color distortion, flickering, and overexposure.\n"
    "Motion & Interaction: whether the subject's motion is natural, smooth, and realistic; "
    "whether interactions among subjects and/or objects are physically plausible; "
    "and whether causal relationships are correctly depicted.\n"
    "Prompt Alignment: whether the subject and object described in the prompt appear accurately, "
    "and whether the subject-object interaction described in the prompt is correctly represented.\n"
    "Prompt: {prompt} Provide your reasoning trace between think tags <think> and </think>, "
    'then output "Yes" or "No" for each dimension between <answer> and </answer>.'
)


class StopSequenceCriteria(StoppingCriteria):
    def __init__(self, stop_sequences: list[list[int]]):
        self.stop_sequences = [seq for seq in stop_sequences if seq]

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        for seq in self.stop_sequences:
            seq_len = len(seq)
            if input_ids.shape[-1] < seq_len:
                continue
            stop_ids = torch.tensor(seq, device=input_ids.device, dtype=input_ids.dtype)
            matches = input_ids[:, -seq_len:].eq(stop_ids).all(dim=-1)
            if matches.all():
                return True
        return False


def first_int(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def infer_model_type(model_path: str) -> str:
    name = str(model_path).lower()
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_model_type = str(getattr(config, "model_type", "")).lower()
        architectures = " ".join(getattr(config, "architectures", []) or []).lower()
        haystack = f"{config_model_type} {architectures} {name}"
    except Exception:
        haystack = name

    if "internvl" in haystack:
        return "internvl"
    if "molmo2" in haystack or "molmo" in haystack:
        return "molmo2"
    if "gemma" in haystack:
        return "gemma4"
    if "minicpm" in haystack:
        return "minicpmv"
    if "qwen2.5" in haystack or "qwen2_5" in haystack:
        return "qwen2.5vl"
    if "qwen2" in haystack:
        return "qwen2vl"
    return "qwen3vl"


def is_hf_internvl_checkpoint(model_path: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return False
    architectures = getattr(config, "architectures", []) or []
    return (
        str(getattr(config, "model_type", "")).lower() == "internvl"
        and any("ForConditionalGeneration" in arch for arch in architectures)
    )


def token_to_id(tokenizer, token: str, required: bool = True) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        ids = tokenizer.encode(token, add_special_tokens=False)
        token_id = ids[0] if len(ids) == 1 else None

    if token_id is None:
        if required:
            raise ValueError(f"Could not resolve tokenizer id for InternVL token: {token}")
        return None
    return int(token_id)


def patch_internvl_tokenizer(tokenizer):
    required_token_attrs = {
        "start_image_token": "<img>",
        "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>",
    }
    optional_token_attrs = {
        "image_token": "<image>",
        "video_token": "<video>",
    }
    for attr, token in {**required_token_attrs, **optional_token_attrs}.items():
        value = getattr(tokenizer, attr, None) or token
        setattr(tokenizer, attr, value)

    tokenizer.start_image_token_id = token_to_id(
        tokenizer,
        tokenizer.start_image_token,
    )
    tokenizer.end_image_token_id = token_to_id(tokenizer, tokenizer.end_image_token)
    tokenizer.context_image_token_id = token_to_id(
        tokenizer,
        tokenizer.context_image_token,
    )
    for attr in optional_token_attrs:
        token_id = token_to_id(tokenizer, getattr(tokenizer, attr), required=False)
        if token_id is not None:
            setattr(tokenizer, f"{attr}_id", token_id)
    return tokenizer


def load_processor(model_path: str, model_type: str):
    processor_kwargs = {
        "trust_remote_code": model_type in {"internvl", "minicpmv", "molmo2"}
    }
    if model_type != "internvl":
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
        if model_type == "molmo2" and not hasattr(processor, "audio_tokenizer"):
            processor.audio_tokenizer = None
        return processor

    try:
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    except AttributeError as exc:
        if "start_image_token" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = patch_internvl_tokenizer(tokenizer)
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                tokenizer=tokenizer,
                **processor_kwargs,
            )
        except AttributeError as second_exc:
            if "start_image_token" not in str(second_exc):
                raise
            from transformers.models.internvl.processing_internvl import InternVLProcessor

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            video_processor = AutoVideoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = InternVLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                image_seq_length=int(getattr(config, "image_seq_length", 256)),
            )

    processor.tokenizer = patch_internvl_tokenizer(processor.tokenizer)
    return processor


def load_model(
    model_path: str,
    model_type: str,
    dtype: str,
    attn_implementation: str | None,
    device_map: str | None = "auto",
):
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": model_type in {"internvl", "minicpmv", "molmo2"},
    }
    if device_map not in {None, "", "none"}:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if model_type == "internvl" and not is_hf_internvl_checkpoint(model_path):
        model = AutoModel.from_pretrained(model_path, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs)

    if model_type == "molmo2":
        patch_molmo2_vision_pooling_device(model)

    return model.eval()


def _first_nested_tensor(value: Any):
    if torch.is_tensor(value):
        return value
    if isinstance(value, dict):
        for nested_value in value.values():
            tensor = _first_nested_tensor(nested_value)
            if tensor is not None:
                return tensor
    if isinstance(value, (list, tuple)):
        for nested_value in value:
            tensor = _first_nested_tensor(nested_value)
            if tensor is not None:
                return tensor
    return None


def _module_tensor_device(module) -> torch.device | None:
    for tensors_fn in (module.parameters, module.buffers):
        for tensor in tensors_fn(recurse=True):
            if tensor.device.type != "meta":
                return tensor.device

    device = getattr(module, "device", None)
    if device is None:
        return None
    return torch.device(device)


def _move_nested_tensors_to_device(value: Any, device: torch.device):
    if torch.is_tensor(value):
        if value.device == device:
            return value
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {
            key: _move_nested_tensors_to_device(nested_value, device)
            for key, nested_value in value.items()
        }
    if isinstance(value, tuple):
        return tuple(_move_nested_tensors_to_device(v, device) for v in value)
    if isinstance(value, list):
        return [_move_nested_tensors_to_device(v, device) for v in value]
    return value


def patch_molmo2_vision_pooling_device(model) -> None:
    core_model = getattr(model, "model", None)
    vision_backbone = getattr(core_model, "vision_backbone", None)
    if vision_backbone is None or getattr(
        vision_backbone,
        "_vlm_reward_pooling_device_patched",
        False,
    ):
        return

    forward_attr = (
        "_old_forward" if hasattr(vision_backbone, "_old_forward") else "forward"
    )
    original_forward = getattr(vision_backbone, forward_attr)

    @functools.wraps(original_forward)
    def patched_forward(*args, **kwargs):
        target_device = _module_tensor_device(vision_backbone)
        if target_device is None:
            tensor = _first_nested_tensor(args)
            if tensor is None:
                tensor = _first_nested_tensor(kwargs)
            target_device = tensor.device if tensor is not None else None

        if target_device is None or target_device.type == "meta":
            return original_forward(*args, **kwargs)

        moved_args = tuple(
            _move_nested_tensors_to_device(arg, target_device) for arg in args
        )
        moved_kwargs = {
            key: _move_nested_tensors_to_device(value, target_device)
            for key, value in kwargs.items()
        }
        return original_forward(*moved_args, **moved_kwargs)

    setattr(vision_backbone, forward_attr, patched_forward)
    vision_backbone._vlm_reward_pooling_device_patched = True


def prepare_processor(processor, model, model_type: str, model_max_length: int):
    tokenizer = processor.tokenizer
    if model_type == "internvl":
        tokenizer = patch_internvl_tokenizer(tokenizer)
        processor.tokenizer = tokenizer

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_max_length

    if model_type == "internvl" and hasattr(model, "img_context_token_id"):
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.context_image_token
        )


def configure_internvl_processor(processor, model, args):
    vision_config = getattr(model.config, "vision_config", None)
    model_image_size = first_int(getattr(vision_config, "image_size", None))
    image_size = int(args.internvl_image_size or model_image_size or 448)
    patch_size = int(first_int(getattr(vision_config, "patch_size", None)) or 14)
    downsample_ratio = float(getattr(model.config, "downsample_ratio", 0.5))

    if image_size % patch_size != 0:
        raise ValueError(
            f"InternVL image size {image_size} must be divisible by patch size {patch_size}."
        )

    grid_size = image_size // patch_size
    pooled_grid_size = grid_size * downsample_ratio
    rounded_grid_size = int(round(pooled_grid_size))
    if abs(pooled_grid_size - rounded_grid_size) > 1e-6:
        raise ValueError(
            f"InternVL image size {image_size} gives patch grid {grid_size}, "
            f"which is incompatible with downsample_ratio {downsample_ratio}."
        )

    if hasattr(processor, "image_seq_length"):
        processor.image_seq_length = rounded_grid_size * rounded_grid_size

    for component in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if component is not None and hasattr(component, "size"):
            component.size = {"height": image_size, "width": image_size}

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_patches"):
            image_processor.min_patches = int(args.internvl_min_patches)
        if hasattr(image_processor, "max_patches"):
            image_processor.max_patches = int(args.internvl_max_patches)

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        if hasattr(video_processor, "num_frames"):
            video_processor.num_frames = int(args.video_max_frames)
        if hasattr(video_processor, "do_sample_frames"):
            video_processor.do_sample_frames = True
        if hasattr(video_processor, "fps"):
            video_processor.fps = None


def build_video_content_kwargs(model_type: str, args) -> dict:
    return {}


def build_messages(
    video_path: str,
    prompt: str,
    model_type: str = "",
    video_content_kwargs: dict | None = None,
):
    user_input = QUESTION_TEMPLATE.format(prompt=prompt)
    video_content = {"type": "video", "video": str(Path(video_path).resolve())}
    video_content.update(video_content_kwargs or {})
    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": user_input},
            ],
        }
    ]


def normalize_molmo2_messages(messages):
    normalized = []
    for message in messages:
        if message.get("role") != "user" or not isinstance(message.get("content"), list):
            normalized.append(message)
            continue
        text_parts = [part for part in message["content"] if part.get("type") == "text"]
        other_parts = [part for part in message["content"] if part.get("type") != "text"]
        normalized.append({**message, "content": text_parts + other_parts})
    return normalized


def get_metadata_value(metadata, key: str):
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def maybe_add_qwen_time_instruction(messages, processor):
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return

    if not hasattr(video_processor, "temporal_patch_size") or not hasattr(
        video_processor, "fps"
    ):
        return

    video_path = messages[0]["content"][0].get("video")
    videos = [video_path] if isinstance(video_path, str) else video_path
    if not videos:
        return

    vp_output = video_processor(videos=videos, return_metadata=True)
    video_grid_thw = getattr(vp_output, "video_grid_thw", None)
    video_metadata = getattr(vp_output, "video_metadata", None)
    if video_grid_thw is None or not video_metadata:
        return

    sample_fps = video_processor.fps
    temporal_patch_size = video_processor.temporal_patch_size
    total_frames = int(video_grid_thw[0][0] * temporal_patch_size)
    duration = get_metadata_value(video_metadata[0], "duration")
    if duration is None:
        return

    time_instruction = (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains "
        f"{total_frames} frames from 0 seconds to {float(duration):.1f} seconds."
    )
    text_content = messages[0]["content"][1]
    text_content["text"] = f"{time_instruction}\n{text_content['text']}"


def build_template_kwargs(model_type: str, args):
    if model_type != "internvl":
        return {}

    return {
        "num_frames": int(args.video_max_frames),
        "do_sample_frames": True,
        "images_kwargs": {
            "min_patches": int(args.internvl_min_patches),
            "max_patches": int(args.internvl_max_patches),
        },
    }


def token_ids_for_text(tokenizer, text: str) -> list[int]:
    token_ids = tokenizer.encode(text, add_special_tokens=False)
    return token_ids if token_ids else []


def collect_eos_token_ids(tokenizer) -> list[int]:
    eos_ids: list[int] = []
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_ids.append(eos_token_id)
    elif isinstance(eos_token_id, (list, tuple)):
        eos_ids.extend(token_id for token_id in eos_token_id if isinstance(token_id, int))

    for token in ("<|im_end|>", "</s>", getattr(tokenizer, "eos_token", None)):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != getattr(tokenizer, "unk_token_id", None):
            eos_ids.append(token_id)

    return sorted(set(eos_ids))


def build_stopping_criteria(tokenizer) -> StoppingCriteriaList:
    stop_sequences = [
        token_ids_for_text(tokenizer, "</answer>"),
        token_ids_for_text(tokenizer, "<|im_end|>"),
    ]
    return StoppingCriteriaList([StopSequenceCriteria(stop_sequences)])


def build_generation_kwargs(tokenizer, args, model_type: str = ""):
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
        "stopping_criteria": build_stopping_criteria(tokenizer),
    }
    if model_type != "molmo2":
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size

    eos_token_ids = collect_eos_token_ids(tokenizer)
    if eos_token_ids:
        generation_kwargs["eos_token_id"] = eos_token_ids

    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p

    return generation_kwargs


def trim_repeated_response(text: str) -> str:
    answer_end = text.find("</answer>")
    if answer_end >= 0:
        return text[: answer_end + len("</answer>")].strip()

    first_think_end = text.find("</think>")
    second_think_start = text.find("<think>", first_think_end + len("</think>"))
    if first_think_end >= 0 and second_think_start >= 0:
        return text[:second_think_start].strip()

    dangling_think = text.find("\n</think>", first_think_end + len("</think>"))
    if first_think_end >= 0 and dangling_think >= 0:
        return text[:dangling_think].strip()

    return text.strip()
