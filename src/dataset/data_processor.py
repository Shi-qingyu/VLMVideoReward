import json
import random
import logging
import re
import time
import itertools
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"
THINK_START_TAG = "<think>"
THINK_END_TAG = "</think>"
ANSWER_START_TAG = "<answer>"
ANSWER_END_TAG = "</answer>"
MEDIA_PLACEHOLDER_PATTERN = re.compile(r"(<image>|<video>)")
THINK_CONTENT_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def _normalize_media_list(media):
    if media is None:
        return []
    if isinstance(media, str):
        return [media]
    return list(media)


def _resolve_media_paths(source: Dict[str, Any]) -> tuple[List[str], List[str]]:
    base_path = Path(source.get("data_path", ""))
    images = [
        _make_abs_paths(base_path, img)
        for img in _normalize_media_list(source.get("images"))
    ]
    videos = [
        _make_abs_paths(base_path, vid)
        for vid in _normalize_media_list(source.get("videos"))
    ]
    return images, videos


def _get_tag_token_ids(tokenizer, tag: str) -> List[int]:
    cache = getattr(tokenizer, "_special_tag_token_ids", None)
    if cache is None:
        cache = {}
        setattr(tokenizer, "_special_tag_token_ids", cache)

    if tag not in cache:
        token_ids = tokenizer.encode(tag, add_special_tokens=False)
        if not token_ids:
            raise ValueError(f"Failed to tokenize special tag: {tag}")
        cache[tag] = token_ids
    return cache[tag]


def _match_token_sequence(tokens: List[int], pos: int, pattern: List[int]) -> bool:
    end = pos + len(pattern)
    return end <= len(tokens) and tokens[pos:end] == pattern


def _find_token_sequence(tokens: List[int], pattern: List[int], start: int) -> int:
    max_start = len(tokens) - len(pattern)
    for pos in range(start, max_start + 1):
        if _match_token_sequence(tokens, pos, pattern):
            return pos
    return -1


def _is_whitespace_token(tokenizer, token_id: int) -> bool:
    cache = getattr(tokenizer, "_whitespace_token_cache", None)
    if cache is None:
        cache = {}
        setattr(tokenizer, "_whitespace_token_cache", cache)
    if token_id not in cache:
        decoded = tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        cache[token_id] = decoded.strip() == ""
    return cache[token_id]


def _is_stop_supervision_token(tokenizer, token_id: int) -> bool:
    special_ids = getattr(tokenizer, "all_special_ids", [])
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if eos_token_id is not None and token_id == eos_token_id:
        return True
    return token_id in special_ids


def _extend_supervision_boundary(tokenizer, tokens: List[int], label_end: int) -> int:
    while label_end < len(tokens) and _is_whitespace_token(tokenizer, tokens[label_end]):
        label_end += 1
    if label_end < len(tokens) and _is_stop_supervision_token(tokenizer, tokens[label_end]):
        label_end += 1
    return label_end


def _build_video_time_instruction(base_path: Path, videos: List[str], processor) -> str:
    video_processor = getattr(processor, "video_processor", None)
    if not videos or video_processor is None:
        return ""
    if not hasattr(video_processor, "fps") or not hasattr(
        video_processor, "temporal_patch_size"
    ):
        return ""

    sample_fps = video_processor.fps
    temporal_patch_size = video_processor.temporal_patch_size
    video_pool = [_make_abs_paths(base_path, vid) for vid in videos]
    vp_output = video_processor(videos=video_pool, return_metadata=True)
    video_metadata = vp_output.video_metadata[0]
    video_grid_thw = vp_output.video_grid_thw

    total_frames = int(video_grid_thw[0][0] * temporal_patch_size)
    duration = video_metadata["duration"]
    return (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames} frames "
        f"from 0 seconds to {duration:.1f} seconds."
    )


def _load_rgb_image(image_path: str) -> Image.Image:
    with Image.open(image_path) as image:
        return image.convert("RGB")


def _sample_frame_indices(total_frames: int, max_frames: Optional[int]) -> List[int]:
    if total_frames <= 0:
        raise ValueError(f"Video contains no frames, got total_frames={total_frames}")
    frame_count = min(int(max_frames or total_frames), total_frames)
    if frame_count <= 1:
        return [0]
    return (
        torch.linspace(0, total_frames - 1, steps=frame_count)
        .round()
        .to(torch.long)
        .tolist()
    )


def _load_video_frames(video_path: str, max_frames: Optional[int]) -> List[Image.Image]:
    video_error: Optional[Exception] = None
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        indices = _sample_frame_indices(len(reader), max_frames)
        frames = reader.get_batch(indices).asnumpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]
    except Exception as exc:
        video_error = exc

    try:
        from torchvision.io import read_video

        video, _, _ = read_video(video_path, pts_unit="sec")
        indices = torch.as_tensor(
            _sample_frame_indices(int(video.shape[0]), max_frames),
            dtype=torch.long,
        )
        frames = video.index_select(0, indices).cpu().numpy()
        return [Image.fromarray(frame).convert("RGB") for frame in frames]
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load video {video_path} with decord and torchvision."
        ) from (video_error or exc)


def _make_temporal_groups(num_frames: int, group_size: int) -> List[List[int]]:
    group_size = max(int(group_size), 1)
    return [
        list(range(start, min(start + group_size, num_frames)))
        for start in range(0, num_frames, group_size)
    ]


def _ensure_batched_input_ids(input_ids: Any) -> torch.Tensor:
    if isinstance(input_ids, torch.Tensor):
        return input_ids.long()
    return torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)


def _add_response_labels(full_result: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    input_ids = _ensure_batched_input_ids(full_result["input_ids"])
    labels = torch.full_like(input_ids, IGNORE_INDEX)

    think_start_ids = _get_tag_token_ids(tokenizer, THINK_START_TAG)
    think_end_ids = _get_tag_token_ids(tokenizer, THINK_END_TAG)
    answer_start_ids = _get_tag_token_ids(tokenizer, ANSWER_START_TAG)
    answer_end_ids = _get_tag_token_ids(tokenizer, ANSWER_END_TAG)
    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if _match_token_sequence(input_ids_flat, pos, think_start_ids):
            think_end = _find_token_sequence(
                input_ids_flat,
                think_end_ids,
                pos + len(think_start_ids),
            )
            if think_end != -1:
                label_end = think_end + len(think_end_ids)
                answer_start = _find_token_sequence(
                    input_ids_flat,
                    answer_start_ids,
                    label_end,
                )
                if answer_start != -1:
                    answer_end = _find_token_sequence(
                        input_ids_flat,
                        answer_end_ids,
                        answer_start + len(answer_start_ids),
                    )
                    if answer_end != -1:
                        label_end = answer_end + len(answer_end_ids)
                label_end = _extend_supervision_boundary(
                    tokenizer,
                    input_ids_flat,
                    label_end,
                )
                labels[0, pos:label_end] = input_ids[0, pos:label_end]
                pos = label_end
                continue

        if _match_token_sequence(input_ids_flat, pos, answer_start_ids):
            answer_end = _find_token_sequence(
                input_ids_flat,
                answer_end_ids,
                pos + len(answer_start_ids),
            )
            if answer_end != -1:
                label_end = _extend_supervision_boundary(
                    tokenizer,
                    input_ids_flat,
                    answer_end + len(answer_end_ids),
                )
                labels[0, pos:label_end] = input_ids[0, pos:label_end]
                pos = label_end
                continue
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def _prepare_shared_qwen_visual_inputs(
    messages: List[Dict[str, Any]],
    processor,
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    from qwen_vl_utils import process_vision_info

    image_patch_size = int(getattr(processor.image_processor, "patch_size", 14))
    rendered_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=image_patch_size,
        return_video_kwargs=True,
        return_video_metadata=True,
    )

    video_metadatas = []
    videos = None
    if video_inputs is not None:
        videos, video_metadatas = zip(*video_inputs)
        videos = list(videos)
        video_metadatas = list(video_metadatas)

    processor_kwargs = {
        "text": [rendered_text],
        "return_tensors": "pt",
        "do_resize": False,
        **video_kwargs,
    }
    if image_inputs is not None:
        processor_kwargs["images"] = image_inputs
    if videos is not None:
        processor_kwargs["videos"] = videos
    if video_metadatas:
        processor_kwargs["video_metadata"] = video_metadatas

    full_result = processor(**processor_kwargs)
    return full_result, video_metadatas


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)
    is_gemma4 = getattr(data_args, "model_type", "") == "gemma4"

    def _update_edge_size(component, shortest_edge: int, longest_edge: int, label: str):
        if not hasattr(component, "size") or not isinstance(component.size, dict):
            return

        size = component.size
        keys = set(size)
        if {"height", "width"}.issubset(keys):
            component.size = {"height": size["height"], "width": size["width"]}
            rank0_print(
                f"Kept {label}.size as height/width: {component.size}"
            )
            return
        if {"max_height", "max_width"}.issubset(keys):
            component.size = {
                "max_height": size["max_height"],
                "max_width": size["max_width"],
            }
            rank0_print(
                f"Kept {label}.size as max_height/max_width: {component.size}"
            )
            return
        if "shortest_edge" in keys or "longest_edge" in keys:
            new_size = {}
            if "shortest_edge" in keys:
                new_size["shortest_edge"] = shortest_edge
            if "longest_edge" in keys:
                new_size["longest_edge"] = longest_edge
            component.size = new_size
            rank0_print(f"Updated {label}.size to {component.size}")

    # --- Image Processor ---
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return processor

    rank0_print("=== BEFORE IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    image_size = getattr(ip, "size", {})
    rank0_print(f"ip.size: {image_size}")
    if isinstance(image_size, dict):
        rank0_print(f"Image size (shortest_edge): {image_size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Image size (longest_edge):  {image_size.get('longest_edge', 'N/A')}")
    rank0_print(f"Image max_soft_tokens: {getattr(ip, 'max_soft_tokens', 'N/A')}")

    if not is_gemma4 and hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if not is_gemma4:
        _update_edge_size(
            ip,
            shortest_edge=data_args.min_pixels,
            longest_edge=data_args.max_pixels,
            label="image_processor",
        )

    gemma4_max_soft_tokens = getattr(data_args, "gemma4_max_soft_tokens", None)
    if gemma4_max_soft_tokens is not None and hasattr(ip, "max_soft_tokens"):
        ip.max_soft_tokens = int(gemma4_max_soft_tokens)
        rank0_print(
            f"Updated image_processor max_soft_tokens to {gemma4_max_soft_tokens}"
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")
    rank0_print(f"Image max_soft_tokens: {getattr(ip, 'max_soft_tokens', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        video_size = getattr(vp, "size", {})
        if isinstance(video_size, dict):
            rank0_print(
                f"Video size (shortest_edge): {video_size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(f"Video size (longest_edge):  {video_size.get('longest_edge', 'N/A')}")
        rank0_print(f"Video max_soft_tokens: {getattr(vp, 'max_soft_tokens', 'N/A')}")

        if not is_gemma4 and hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
            vp.min_pixels = data_args.video_min_pixels
            vp.max_pixels = data_args.video_max_pixels
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor min_pixels to {data_args.video_min_pixels}"
            )
            rank0_print(
                f"✅ Updated Qwen2-VL video_processor max_pixels to {data_args.video_max_pixels}"
            )

        if hasattr(vp, "min_frames") and hasattr(vp, "max_frames"):
            vp.min_frames = data_args.video_min_frames
            vp.max_frames = data_args.video_max_frames
            rank0_print(
                f"✅ Updated video_processor min_frames to {data_args.video_min_frames}"
            )
            rank0_print(
                f"✅ Updated video_processor max_frames to {data_args.video_max_frames}"
            )

        if hasattr(vp, "fps"):
            vp.fps = data_args.video_fps
            rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if not is_gemma4:
            _update_edge_size(
                vp,
                shortest_edge=data_args.video_min_pixels,
                longest_edge=data_args.video_max_pixels,
                label="video_processor",
            )

        if gemma4_max_soft_tokens is not None and hasattr(vp, "max_soft_tokens"):
            vp.max_soft_tokens = int(gemma4_max_soft_tokens)
            rank0_print(
                f"Updated video_processor max_soft_tokens to {gemma4_max_soft_tokens}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            rank0_print(
                f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")
        rank0_print(f"Video max_soft_tokens: {getattr(vp, 'max_soft_tokens', 'N/A')}")

    return processor


def _build_messages(item: Dict[str, Any], base_path: Path, using_cot: bool = True, time_instruction: str = "") -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = _normalize_media_list(item.get("images"))
    videos = _normalize_media_list(item.get("videos"))

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_pool = [
        {"type": "video", "video": _make_abs_paths(base_path, vid)} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = MEDIA_PLACEHOLDER_PATTERN.split(text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    if time_instruction:
                        seg = f"{time_instruction}\n{seg.strip()}"
                        time_instruction = ""
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            # Assistant messages contain only text
            if not using_cot:
                text = THINK_CONTENT_PATTERN.sub("", text)
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
    using_cot: bool = True,
    share_distill_video_sampling: bool = False,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))

    videos = _normalize_media_list(source.get("videos"))
    time_instruction = _build_video_time_instruction(base_path, videos, processor)

    messages = _build_messages(source, base_path, using_cot, time_instruction)
    video_metadatas = []
    if share_distill_video_sampling:
        try:
            full_result, video_metadatas = _prepare_shared_qwen_visual_inputs(
                messages,
                processor,
            )
        except Exception as exc:
            logging.getLogger(__name__).warning(
                "Falling back to processor.apply_chat_template for visual inputs because shared Qwen video sampling failed: %s",
                exc,
            )
            full_result = processor.apply_chat_template(
                messages, tokenize=True, return_dict=True, return_tensors="pt"
            )
    else:
        full_result = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )

    full_result = _add_response_labels(full_result, processor.tokenizer)
    full_result["distill_video_metadatas"] = video_metadatas
    return full_result


def preprocess_gemma4_visual(
    sources,
    processor,
    using_cot: bool = True,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path, using_cot, time_instruction="")
    full_result = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    full_result = _add_response_labels(full_result, processor.tokenizer)
    full_result["distill_video_metadatas"] = []
    return full_result


def preprocess_hf_chat_visual(
    sources,
    processor,
    using_cot: bool = True,
) -> Dict:
    return preprocess_gemma4_visual(sources, processor, using_cot)


def _build_minicpmv_messages_and_images(
    item: Dict[str, Any],
    base_path: Path,
    using_cot: bool,
    max_video_frames: Optional[int],
    video_group_size: int,
) -> tuple[List[Dict[str, str]], List[Image.Image], List[List[int]]]:
    images = _normalize_media_list(item.get("images"))
    videos = _normalize_media_list(item.get("videos"))

    image_pool = [_make_abs_paths(base_path, img) for img in images]
    video_pool = [_make_abs_paths(base_path, vid) for vid in videos]

    temporal_groups: List[List[int]] = []
    input_images: List[Image.Image] = []
    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content_parts = []
            for seg in MEDIA_PLACEHOLDER_PATTERN.split(text):
                if seg == DEFAULT_IMAGE_TOKEN:
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content_parts.append("<image>./</image>")
                    input_images.append(_load_rgb_image(image_pool.pop(0)))
                    temporal_groups.append([-1])
                elif seg == DEFAULT_VIDEO_TOKEN:
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    frames = _load_video_frames(video_pool.pop(0), max_video_frames)
                    if not frames:
                        raise ValueError("Loaded video has no frames")
                    frame_placeholders = []
                    start_idx = len(input_images)
                    for frame_idx, _frame in enumerate(frames):
                        frame_placeholders.append(f"Frame{frame_idx + 1}: <image>./</image>")
                        input_images.append(_frame)
                    temporal_groups.extend(
                        [
                            [start_idx + frame_idx for frame_idx in group]
                            for group in _make_temporal_groups(
                                len(frames),
                                video_group_size,
                            )
                        ]
                    )
                    content_parts.append("\n".join(frame_placeholders))
                elif seg.strip():
                    content_parts.append(seg.strip())

            messages.append({"role": role, "content": "\n".join(content_parts)})
        else:
            if not using_cot:
                text = THINK_CONTENT_PATTERN.sub("", text)
            messages.append({"role": role, "content": text})

    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages, input_images, temporal_groups


def preprocess_minicpmv_visual(
    sources,
    processor,
    data_args,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages, input_images, temporal_groups = _build_minicpmv_messages_and_images(
        source,
        base_path,
        getattr(data_args, "using_cot", True),
        getattr(data_args, "video_max_frames", None),
        getattr(data_args, "minicpmv_video_group_size", 6),
    )
    rendered_text = processor.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    processor_kwargs = {
        "text": [rendered_text],
        "images": [input_images],
        "return_tensors": "pt",
        "max_length": getattr(processor.tokenizer, "model_max_length", None),
    }
    max_slice_nums = getattr(data_args, "minicpmv_max_slice_nums", None)
    if max_slice_nums is not None:
        processor_kwargs["max_slice_nums"] = max_slice_nums
    if temporal_groups:
        processor_kwargs["temporal_ids"] = [temporal_groups]

    full_result = processor(**processor_kwargs)
    full_result = _add_response_labels(full_result, processor.tokenizer)
    attention_mask = full_result.get("attention_mask")
    if attention_mask is None:
        attention_mask = full_result["input_ids"].ne(processor.tokenizer.pad_token_id)
        full_result["attention_mask"] = attention_mask
    position_ids = attention_mask.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention_mask.eq(0), 0)
    full_result["position_ids"] = position_ids
    full_result["distill_video_metadatas"] = []
    full_result["_model_type"] = "minicpmv"
    return full_result


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, processor, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.video_max_total_pixels = getattr(
            data_args, "video_max_total_pixels", 1664 * 28 * 28
        )
        self.video_min_total_pixels = getattr(
            data_args, "video_min_total_pixels", 256 * 28 * 28
        )
        self.model_type = getattr(data_args, "model_type", "qwen3vl")
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        elif data_args.model_type in {"gemma4", "internvl", "minicpmv"}:
            self.get_rope_index = None
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                else:
                    ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")


        rank0_print("Formatting inputs...Skip in lazy mode")
        processor = update_processor_pixels(processor, data_args)
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.data_args = data_args
        self.merge_size = getattr(
            getattr(processor, "image_processor", None),
            "merge_size",
            1,
        )
        self.list_data_dict = list_data_dict

        if data_args.data_packing:
            self.item_fn = self._get_packed_item
        else:
            self.item_fn = self._get_item

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    @property
    def pre_calculated_length(self):
        if "num_tokens" in self.list_data_dict[0]:
            length_list = [sample["num_tokens"] for sample in self.list_data_dict]
            return np.array(length_list)
        else:
            print("No pre-calculated length available.")
            return np.array([1] * len(self.list_data_dict))

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3

        # try the current sample first
        for attempt_idx in range(num_base_retries):
            try:
                sources = self.list_data_dict[i]
                if isinstance(sources, dict):
                    sources = [sources]
                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # sleep 1s in case it is a cloud disk issue
                print(f"[Try #{attempt_idx}] Failed to fetch sample {i}. Exception:", e)
                time.sleep(1)

        # try other samples, in case it is file corruption issue
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sources = self.list_data_dict[next_index]
                if isinstance(sources, dict):
                    sources = [sources]

                sample = self.item_fn(sources)
                return sample
            except Exception as e:
                # no need to sleep
                print(
                    f"[Try other #{attempt_idx}] Failed to fetch sample {next_index}. Exception:",
                    e,
                )
                pass

        try:
            sources = self.list_data_dict[i]
            if isinstance(sources, dict):
                sources = [sources]
            sample = self.item_fn(sources)
            return sample
        except Exception as e:
            raise e

    def _get_item(self, sources) -> Dict[str, torch.Tensor]:
        if self.model_type in {"gemma4", "internvl"}:
            data_dict = preprocess_hf_chat_visual(
                sources,
                self.processor,
                self.data_args.using_cot,
            )
            if (
                self.model_type == "internvl"
                and getattr(self.data_args, "internvl_use_image_flags", False)
                and "pixel_values" in data_dict
            ):
                data_dict["image_flags"] = torch.ones(
                    data_dict["pixel_values"].shape[0],
                    1,
                    dtype=torch.long,
                )
            image_paths, video_paths = _resolve_media_paths(sources[0])
            data_dict["distill_image_paths"] = image_paths
            data_dict["distill_video_paths"] = video_paths
            data_dict["distill_video_metadatas"] = data_dict.get(
                "distill_video_metadatas", []
            )
            return data_dict

        if self.model_type == "minicpmv":
            data_dict = preprocess_minicpmv_visual(
                sources,
                self.processor,
                self.data_args,
            )
            image_paths, video_paths = _resolve_media_paths(sources[0])
            data_dict["distill_image_paths"] = image_paths
            data_dict["distill_video_paths"] = video_paths
            data_dict["distill_video_metadatas"] = data_dict.get(
                "distill_video_metadatas", []
            )
            return data_dict

        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            self.data_args.using_cot,
            getattr(self.data_args, "distill_share_student_video_sampling", False),
        )
        image_paths, video_paths = _resolve_media_paths(sources[0])

        seq_len = data_dict["input_ids"][0].size(0)

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = [seq_len]
        data_dict["distill_image_paths"] = image_paths
        data_dict["distill_video_paths"] = video_paths
        data_dict["distill_video_metadatas"] = data_dict.get(
            "distill_video_metadatas", []
        )

        return data_dict

    def _get_packed_item(self, sources) -> Dict[str, torch.Tensor]:

        if isinstance(sources, dict):
            sources = [sources]
        if not isinstance(sources, list):
            raise TypeError(f"Unsupported packed source type: {type(sources)}")
        if len(sources) == 1 and isinstance(sources[0], dict):
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            return self._get_item(sources)

        data_list = []
        new_data_dict = {}
        for source in sources:
            if isinstance(source, dict):
                source = [source]
            assert (
                len(source) == 1
            ), f"Don't know why it is wrapped to a list.\n {source}"  # FIXME
            data_list.append(self._get_item(source))

        input_ids = torch.cat([d["input_ids"] for d in data_list], dim=1)
        labels = torch.cat([d["labels"] for d in data_list], dim=1)
        position_ids = torch.cat([d["position_ids"] for d in data_list], dim=2)
        attention_mask = [
            d["attention_mask"][0] for d in data_list if "attention_mask" in d
        ]
        new_data_dict = {
            "input_ids": input_ids,
            "labels": labels,
            "position_ids": position_ids,
            "attention_mask": attention_mask if attention_mask else None,
            "distill_image_paths": list(
                itertools.chain.from_iterable(
                    d.get("distill_image_paths", []) for d in data_list
                )
            ),
            "distill_video_paths": list(
                itertools.chain.from_iterable(
                    d.get("distill_video_paths", []) for d in data_list
                )
            ),
            "distill_video_metadatas": list(
                itertools.chain.from_iterable(
                    d.get("distill_video_metadatas", []) for d in data_list
                )
            ),
        }

        if any("pixel_values" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values": torch.cat(
                        [
                            d["pixel_values"]
                            for d in data_list
                            if "pixel_values" in d
                        ],
                        dim=0,
                    ),
                    "image_grid_thw": torch.cat(
                        [
                            d["image_grid_thw"]
                            for d in data_list
                            if "image_grid_thw" in d
                        ],
                        dim=0,
                    ),
                }
            )

        if any("pixel_values_videos" in d for d in data_list):
            new_data_dict.update(
                {
                    "pixel_values_videos": torch.cat(
                        [
                            d["pixel_values_videos"]
                            for d in data_list
                            if "pixel_values_videos" in d
                        ],
                        dim=0,
                    ),
                    "video_grid_thw": torch.cat(
                        [
                            d["video_grid_thw"]
                            for d in data_list
                            if "video_grid_thw" in d
                        ],
                        dim=0,
                    ),
                }
            )
        return new_data_dict


def pad_and_cat(tensor_list):
    max_length = max(tensor.shape[2] for tensor in tensor_list)

    padded_tensors = []
    for tensor in tensor_list:
        pad_length = max_length - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", 1)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


def _as_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.as_tensor(value)


def _squeeze_leading_singleton(value: Any) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim > 0 and tensor.shape[0] == 1:
        return tensor.squeeze(0)
    return tensor


def _pad_and_cat_tensors(
    tensors: Sequence[torch.Tensor],
    concat_dim: int = 0,
    pad_value: float = 0,
) -> Optional[torch.Tensor]:
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        return None

    max_shape = list(tensors[0].shape)
    for tensor in tensors[1:]:
        if tensor.ndim != len(max_shape):
            raise ValueError(
                f"Cannot collate tensors with ranks {len(max_shape)} and {tensor.ndim}."
            )
        for dim, size in enumerate(tensor.shape):
            if dim != concat_dim:
                max_shape[dim] = max(max_shape[dim], int(size))

    padded = []
    for tensor in tensors:
        pad_spec = []
        for dim in reversed(range(tensor.ndim)):
            pad_spec.extend([0, 0 if dim == concat_dim else max_shape[dim] - tensor.shape[dim]])
        padded.append(torch.nn.functional.pad(tensor, pad_spec, value=pad_value))
    return torch.cat(padded, dim=concat_dim)


def _collect_tensor_field(
    instances: Sequence[Dict],
    key: str,
    pad_value: float = 0,
) -> Optional[torch.Tensor]:
    tensors = [_as_tensor(instance[key]) for instance in instances if key in instance and instance[key] is not None]
    return _pad_and_cat_tensors(tensors, concat_dim=0, pad_value=pad_value)


def _first_batch_value(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if instances and instances[0].get("_model_type") == "minicpmv":
            return self._collate_minicpmv(instances)

        input_ids, labels = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels")
        )
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if all("position_ids" in instance for instance in instances):
            position_ids = pad_and_cat([instance["position_ids"] for instance in instances])
            batch["position_ids"] = position_ids[:, :, : self.tokenizer.model_max_length]

        if any("mm_token_type_ids" in instance for instance in instances):
            mm_token_type_ids = []
            for instance in instances:
                if "mm_token_type_ids" in instance:
                    token_type_ids = _squeeze_leading_singleton(
                        instance["mm_token_type_ids"]
                    )
                else:
                    token_type_ids = torch.zeros(
                        instance["input_ids"].shape[-1],
                        dtype=input_ids.dtype,
                    )
                mm_token_type_ids.append(token_type_ids)
            batch["mm_token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                mm_token_type_ids,
                batch_first=True,
                padding_value=0,
            )[:, : self.tokenizer.model_max_length]

        for key, pad_value in (
            ("pixel_values", 0),
            ("image_grid_thw", 0),
            ("image_flags", 1),
            ("pixel_values_videos", 0),
            ("video_grid_thw", 0),
            ("image_position_ids", -1),
            ("video_position_ids", -1),
            ("input_features", 0),
            ("input_features_mask", 0),
        ):
            value = _collect_tensor_field(instances, key, pad_value=pad_value)
            if value is not None:
                batch[key] = value

        batch["distill_image_paths"] = [
            instance.get("distill_image_paths", []) for instance in instances
        ]
        batch["distill_video_paths"] = [
            instance.get("distill_video_paths", []) for instance in instances
        ]
        batch["distill_video_metadatas"] = [
            instance.get("distill_video_metadatas", []) for instance in instances
        ]
        return batch

    def _collate_minicpmv(self, instances: Sequence[Dict]) -> Dict[str, Any]:
        input_ids = [
            _squeeze_leading_singleton(instance["input_ids"])
            for instance in instances
        ]
        labels = [
            _squeeze_leading_singleton(instance["labels"])
            for instance in instances
        ]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id)
        position_ids = attention_mask.long().cumsum(dim=-1) - 1
        position_ids.masked_fill_(attention_mask.eq(0), 0)

        data = {
            "input_ids": input_ids,
            "pixel_values": [
                _first_batch_value(instance.get("pixel_values", []))
                for instance in instances
            ],
            "image_bound": [
                _first_batch_value(instance.get("image_bound", []))
                for instance in instances
            ],
            "tgt_sizes": [
                _first_batch_value(instance.get("tgt_sizes", []))
                for instance in instances
            ],
            "temporal_ids": [
                _first_batch_value(instance.get("temporal_ids", []))
                for instance in instances
            ],
            "position_ids": position_ids,
        }

        return {
            "data": data,
            "labels": labels,
            "attention_mask": attention_mask,
            "distill_image_paths": [
                instance.get("distill_image_paths", []) for instance in instances
            ],
            "distill_video_paths": [
                instance.get("distill_video_paths", []) for instance in instances
            ],
            "distill_video_metadatas": [
                instance.get("distill_video_metadatas", []) for instance in instances
            ],
        }


@dataclass
class FlattenedDataCollatorForSupervisedDataset(DataCollatorForSupervisedDataset):
    """Collate examples into packed sequence with multi-modal support."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids, attention_mask = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids", "attention_mask")
        )
        attention_mask = list(
            itertools.chain(
                *(
                    instance["attention_mask"]
                    for instance in instances
                    if "attention_mask" in instance
                )
            )
        )
        seq_lens = torch.tensor([0] + attention_mask, dtype=torch.int32)
        cumsum_seq_lens = torch.cumsum(seq_lens, dim=0, dtype=torch.int32)
        input_ids = torch.cat(input_ids, dim=1)
        labels = torch.cat(labels, dim=1)
        position_ids = torch.cat(position_ids, dim=2)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=cumsum_seq_lens,
            position_ids=position_ids,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["distill_image_paths"] = [
            instance.get("distill_image_paths", []) for instance in instances
        ]
        batch["distill_video_paths"] = [
            instance.get("distill_video_paths", []) for instance in instances
        ]
        batch["distill_video_metadatas"] = [
            instance.get("distill_video_metadatas", []) for instance in instances
        ]

        return batch


class LazyRLDataset(Dataset):
    """Dataset for RL."""

    def __init__(self, processor, data_args):
        super().__init__()
        self.processor = processor

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")
        self.using_cot = getattr(data_args, "using_cot", False)

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))

            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                annotations = random.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                rank0_print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")

            for ann in annotations:
                if isinstance(ann, list):
                    for sub_ann in ann:
                        sub_ann["data_path"] = data["data_path"]
                        list_data_dict.append(sub_ann)
                else:
                    ann["data_path"] = data["data_path"]
                    list_data_dict.append(ann)

        self.list_data_dict = list_data_dict
        rank0_print(f"Total training samples: {len(list_data_dict)}")

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_retries = 3

        for attempt_idx in range(num_retries):
            cur_i = i if attempt_idx == 0 else random.randint(0, len(self.list_data_dict) - 1)
            try:
                source = self.list_data_dict[cur_i]
                base_path = Path(source.get("data_path", ""))

                videos = _normalize_media_list(source.get("videos"))
                time_instruction = _build_video_time_instruction(base_path, videos, self.processor)
                messages = _build_messages(source, base_path, self.using_cot, time_instruction)

                assert len(messages) == 2, f"Expected 2 messages, got {len(messages)}"

                if messages[0]["role"] == "user":
                    user = messages[0]
                    gt = messages[1]
                else:
                    user = messages[1]
                    gt = messages[0]

                _ = self.processor.apply_chat_template(
                    [user],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                    padding=True,
                )

                return {
                    "user": [user],
                    "gt": [gt],
                }

            except Exception as e:
                print(f"[Try #{attempt_idx}] Failed to fetch sample {cur_i}. Exception: {e}")
                time.sleep(1)

        raise RuntimeError(f"Failed to fetch sample {i} after {num_retries} retries")


@dataclass
class DataCollatorForRLDataset(object):
    """Collate examples into packed sequence with multi-modal support."""
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        batch = dict()
        for key in ["user", "gt"]:
            batch[key] = [instance[key] for instance in instances]

        return batch
    

def make_supervised_data_module(processor, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    if getattr(data_args, "model_type", "") in {"gemma4", "internvl", "minicpmv"} and (
        data_args.data_flatten or data_args.data_packing
    ):
        raise ValueError(
            f"{data_args.model_type} training currently uses the standard padded collator. "
            "Set --data_flatten False and --data_packing False."
        )

    train_dataset = LazySupervisedDataset(processor, data_args=data_args)
    if data_args.data_flatten or data_args.data_packing:
        data_collator = FlattenedDataCollatorForSupervisedDataset(processor.tokenizer)
        return dict(
            train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
        )
    data_collator = DataCollatorForSupervisedDataset(processor.tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


def make_rl_data_module(processor, data_args) -> Dict:
    train_dataset = LazyRLDataset(processor, data_args=data_args)
    data_collator = DataCollatorForRLDataset()
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )


if __name__ == "__main__":
    pass
