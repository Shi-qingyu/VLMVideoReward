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
CHAT_END_MARKERS = ["<|im_end|>", "<end_of_turn>", "</s>"]
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


def _find_token_sequence_before(
    tokens: List[int],
    pattern: List[int],
    start: int,
    stop: int,
) -> int:
    if not pattern:
        return -1
    max_start = min(stop, len(tokens)) - len(pattern)
    for pos in range(start, max_start + 1):
        if _match_token_sequence(tokens, pos, pattern):
            return pos
    return -1


def _encode_text_pattern(tokenizer, text: str) -> List[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def _collect_patterns(tokenizer, texts: List[str]) -> List[List[int]]:
    patterns = []
    for text in texts:
        token_ids = _encode_text_pattern(tokenizer, text)
        if token_ids and token_ids not in patterns:
            patterns.append(token_ids)
    return sorted(patterns, key=len, reverse=True)


def _find_first_pattern(
    tokens: List[int],
    patterns: List[List[int]],
    start: int,
    stop: int,
) -> tuple[int, int]:
    best_pos = -1
    best_len = 0
    for pattern in patterns:
        pos = _find_token_sequence_before(tokens, pattern, start, stop)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_len = len(pattern)
    return best_pos, best_len


def _assistant_response_spans(
    tokens: List[int],
    tokenizer,
) -> List[tuple[int, int]]:
    assistant_start_patterns = _collect_patterns(
        tokenizer,
        [
            "<|im_start|>assistant\n",
            "<|im_start|>assistant",
            "<start_of_turn>model\n",
            "<start_of_turn>model",
            "<|assistant|>\n",
            "<|assistant|>",
        ],
    )
    assistant_end_patterns = _collect_patterns(
        tokenizer,
        [
            "<|im_end|>",
            "<end_of_turn>",
            "</s>",
        ],
    )
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        eos_pattern = [eos_token_id]
        if eos_pattern not in assistant_end_patterns:
            assistant_end_patterns.append(eos_pattern)

    if not assistant_start_patterns:
        return [(0, len(tokens))]

    spans: List[tuple[int, int]] = []
    search_pos = 0
    while search_pos < len(tokens):
        start_pos, start_len = _find_first_pattern(
            tokens,
            assistant_start_patterns,
            search_pos,
            len(tokens),
        )
        if start_pos == -1:
            break

        content_start = start_pos + start_len
        end_pos, end_len = _find_first_pattern(
            tokens,
            assistant_end_patterns,
            content_start,
            len(tokens),
        )
        content_end = end_pos if end_pos != -1 else len(tokens)
        if content_start < content_end:
            spans.append((content_start, content_end))
        search_pos = (end_pos + max(end_len, 1)) if end_pos != -1 else len(tokens)

    return spans or [(0, len(tokens))]


def _token_char_offsets(tokens: List[int], tokenizer) -> tuple[str, List[tuple[int, int]]]:
    pieces = [
        tokenizer.decode(
            [token_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        for token_id in tokens
    ]
    offsets: List[tuple[int, int]] = []
    cursor = 0
    for piece in pieces:
        start = cursor
        cursor += len(piece)
        offsets.append((start, cursor))
    return "".join(pieces), offsets


def _find_first_text_marker(
    text: str,
    markers: List[str],
    start: int,
    stop: Optional[int] = None,
) -> tuple[int, str]:
    search_stop = len(text) if stop is None else min(stop, len(text))
    best_pos = -1
    best_marker = ""
    for marker in sorted(markers, key=len, reverse=True):
        pos = text.find(marker, start, search_stop)
        if pos != -1 and (best_pos == -1 or pos < best_pos):
            best_pos = pos
            best_marker = marker
    return best_pos, best_marker


def _assistant_text_spans(text: str) -> List[tuple[int, int]]:
    start_markers = [
        "<|im_start|>assistant\n",
        "<|im_start|>assistant",
        "<start_of_turn>model\n",
        "<start_of_turn>model",
        "<|assistant|>\n",
        "<|assistant|>",
    ]
    spans: List[tuple[int, int]] = []
    search_pos = 0

    while search_pos < len(text):
        start_pos, marker = _find_first_text_marker(text, start_markers, search_pos)
        if start_pos == -1:
            break
        content_start = start_pos + len(marker)
        if content_start < len(text) and text[content_start] == "\n":
            content_start += 1
        end_pos, end_marker = _find_first_text_marker(
            text,
            CHAT_END_MARKERS,
            content_start,
        )
        content_end = end_pos if end_pos != -1 else len(text)
        if content_start < content_end:
            spans.append((content_start, content_end))
        search_pos = (
            end_pos + max(len(end_marker), 1)
            if end_pos != -1
            else len(text)
        )

    return spans or [(0, len(text))]


def _extend_text_label_end(text: str, label_end: int) -> int:
    while label_end < len(text) and text[label_end].isspace():
        label_end += 1
    for marker in sorted(CHAT_END_MARKERS, key=len, reverse=True):
        if text.startswith(marker, label_end):
            return label_end + len(marker)
    return label_end


def _response_text_label_spans(text: str) -> List[tuple[int, int]]:
    label_spans: List[tuple[int, int]] = []

    for span_start, span_end in _assistant_text_spans(text):
        pos = span_start
        while pos < span_end:
            think_start = text.find(THINK_START_TAG, pos, span_end)
            answer_start = text.find(ANSWER_START_TAG, pos, span_end)

            if think_start == -1 and answer_start == -1:
                break

            if think_start != -1 and (
                answer_start == -1 or think_start < answer_start
            ):
                think_end = text.find(
                    THINK_END_TAG,
                    think_start + len(THINK_START_TAG),
                    span_end,
                )
                if think_end == -1:
                    pos = think_start + len(THINK_START_TAG)
                    continue
                label_end = think_end + len(THINK_END_TAG)
                next_answer_start = text.find(ANSWER_START_TAG, label_end, span_end)
                if next_answer_start != -1:
                    answer_end = text.find(
                        ANSWER_END_TAG,
                        next_answer_start + len(ANSWER_START_TAG),
                        span_end,
                    )
                    if answer_end != -1:
                        label_end = answer_end + len(ANSWER_END_TAG)
                label_end = _extend_text_label_end(text, label_end)
                label_spans.append((think_start, label_end))
                pos = label_end
                continue

            answer_end = text.find(
                ANSWER_END_TAG,
                answer_start + len(ANSWER_START_TAG),
                span_end,
            )
            if answer_end == -1:
                pos = answer_start + len(ANSWER_START_TAG)
                continue
            label_end = answer_end + len(ANSWER_END_TAG)
            label_end = _extend_text_label_end(text, label_end)
            label_spans.append((answer_start, label_end))
            pos = label_end

    return label_spans


def _label_tokens_from_char_spans(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    token_offsets: List[tuple[int, int]],
    label_spans: List[tuple[int, int]],
) -> None:
    for token_index, (token_start, token_end) in enumerate(token_offsets):
        if token_start == token_end:
            continue
        for label_start, label_end in label_spans:
            if token_start < label_end and token_end > label_start:
                labels[0, token_index] = input_ids[0, token_index]
                break


def _label_immediate_eos_after_spans(
    labels: torch.Tensor,
    input_ids: torch.Tensor,
    token_offsets: List[tuple[int, int]],
    label_spans: List[tuple[int, int]],
    tokenizer,
) -> None:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        return

    tokens = input_ids[0].tolist()
    for label_start, label_end in label_spans:
        last_labeled_index = -1
        for token_index, (token_start, token_end) in enumerate(token_offsets):
            if token_start < label_end and token_end > label_start:
                last_labeled_index = token_index

        token_index = last_labeled_index + 1
        while token_index < len(tokens) and _is_whitespace_token(
            tokenizer,
            tokens[token_index],
        ):
            token_index += 1

        if token_index < len(tokens) and tokens[token_index] == eos_token_id:
            for index in range(last_labeled_index + 1, token_index + 1):
                labels[0, index] = input_ids[0, index]


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


def _as_positive_float(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _as_positive_int(value) -> Optional[int]:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return None
    return value if value > 0 else None


def _metadata_value(metadata, key: str):
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def _read_video_metadata(video_path: str) -> tuple[Optional[float], Optional[int], Optional[float]]:
    try:
        from decord import VideoReader, cpu

        reader = VideoReader(video_path, ctx=cpu(0))
        source_frames = len(reader)
        source_fps = _as_positive_float(reader.get_avg_fps())
        duration = source_frames / source_fps if source_frames and source_fps else None
        return duration, source_frames, source_fps
    except Exception:
        pass

    try:
        from torchvision.io import read_video_timestamps

        timestamps, source_fps = read_video_timestamps(video_path, pts_unit="sec")
        source_frames = len(timestamps)
        source_fps = _as_positive_float(source_fps)
        duration = None
        if source_frames and source_fps:
            duration = float(timestamps[-1]) + 1.0 / source_fps
        return duration, source_frames, source_fps
    except Exception:
        return None, None, None


def _first_video_processor_fps(video_processor) -> Optional[float]:
    for attr_name in ("fps", "max_fps"):
        fps = _as_positive_float(getattr(video_processor, attr_name, None))
        if fps is not None:
            return fps
    return None


def _processor_video_time_info(video_processor, video_pool: List[str]):
    if video_processor is None or not hasattr(video_processor, "temporal_patch_size"):
        return None, None, None

    temporal_patch_size = _as_positive_int(
        getattr(video_processor, "temporal_patch_size", None)
    )
    if temporal_patch_size is None:
        return None, None, None

    try:
        vp_output = video_processor(videos=video_pool, return_metadata=True)
    except Exception:
        return None, None, None

    video_grid_thw = getattr(vp_output, "video_grid_thw", None)
    video_metadata = getattr(vp_output, "video_metadata", None)
    if video_grid_thw is None:
        return None, None, None

    try:
        grid_t = video_grid_thw[0][0]
        if hasattr(grid_t, "item"):
            grid_t = grid_t.item()
        sampled_frames = int(grid_t * temporal_patch_size)
    except Exception:
        sampled_frames = None

    duration = None
    if video_metadata:
        duration = _as_positive_float(_metadata_value(video_metadata[0], "duration"))

    return _first_video_processor_fps(video_processor), sampled_frames, duration


def _resolve_sampled_video_frames(
    video_processor,
    data_args,
    source_frames: Optional[int],
    duration: Optional[float],
    sample_fps: Optional[float],
) -> Optional[int]:
    model_type = getattr(data_args, "model_type", "")
    for attr_name in ("num_frames", "max_frames"):
        sampled_frames = _as_positive_int(getattr(video_processor, attr_name, None))
        if sampled_frames is not None:
            return min(sampled_frames, source_frames) if source_frames else sampled_frames

    data_args_frame_limit_applies = model_type.startswith("qwen") or model_type in {
        "internvl",
        "minicpmv",
        "molmo2",
    }
    sampled_frames = (
        _as_positive_int(getattr(data_args, "video_max_frames", None))
        if data_args_frame_limit_applies
        else None
    )
    if sampled_frames is not None:
        return min(sampled_frames, source_frames) if source_frames else sampled_frames

    if duration is not None and sample_fps is not None:
        sampled_frames = max(1, int(round(duration * sample_fps)))
        return min(sampled_frames, source_frames) if source_frames else sampled_frames

    return source_frames


def _resolve_sample_fps(
    video_processor,
    data_args,
    duration: Optional[float],
    sampled_frames: Optional[int],
) -> Optional[float]:
    model_type = getattr(data_args, "model_type", "")
    uses_fixed_frame_count = (
        model_type in {"internvl", "minicpmv"}
        and _as_positive_int(getattr(data_args, "video_max_frames", None)) is not None
    )

    if not uses_fixed_frame_count:
        sample_fps = _first_video_processor_fps(video_processor)
        if sample_fps is not None:
            return sample_fps

        sample_fps = _as_positive_float(getattr(data_args, "video_fps", None))
        if sample_fps is not None:
            return sample_fps

    if duration is not None and sampled_frames is not None:
        return sampled_frames / duration

    return None


def _format_video_time_instruction(
    sample_fps: Optional[float],
    sampled_frames: Optional[int],
    duration: Optional[float],
) -> str:
    if sample_fps is None or sampled_frames is None or duration is None:
        return ""
    return (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {sampled_frames} frames "
        f"from 0 seconds to {duration:.1f} seconds."
    )


def _build_video_time_instruction(
    base_path: Path,
    videos: List[str],
    processor,
    data_args=None,
) -> str:
    if not videos:
        return ""

    video_pool = [_make_abs_paths(base_path, vid) for vid in videos]
    video_processor = getattr(processor, "video_processor", None)
    sample_fps, sampled_frames, duration = _processor_video_time_info(
        video_processor,
        video_pool,
    )

    source_frames = None
    if duration is None or sampled_frames is None or sample_fps is None:
        source_duration, source_frames, _source_fps = _read_video_metadata(video_pool[0])
        duration = duration or source_duration

    sampled_frames = sampled_frames or _resolve_sampled_video_frames(
        video_processor,
        data_args,
        source_frames,
        duration,
        sample_fps,
    )
    sample_fps = sample_fps or _resolve_sample_fps(
        video_processor,
        data_args,
        duration,
        sampled_frames,
    )

    if duration is None or duration <= 0:
        return ""
    return _format_video_time_instruction(sample_fps, sampled_frames, duration)


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


def _append_sequence_value(value: Any, fill_value: int, seq_len: int) -> Any:
    if not isinstance(value, torch.Tensor):
        return value
    if value.dim() == 0 or value.shape[-1] != seq_len:
        return value

    suffix = torch.full(
        (*value.shape[:-1], 1),
        fill_value,
        dtype=value.dtype,
        device=value.device,
    )
    return torch.cat([value, suffix], dim=-1)


def _ensure_molmo2_trailing_eos(
    full_result: Dict[str, Any],
    tokenizer,
) -> Dict[str, Any]:
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        return full_result

    input_ids = _ensure_batched_input_ids(full_result["input_ids"])
    if input_ids.numel() == 0:
        full_result["input_ids"] = input_ids
        return full_result

    decoded_text, _ = _token_char_offsets(input_ids[0].tolist(), tokenizer)
    if not _response_text_label_spans(decoded_text):
        full_result["input_ids"] = input_ids
        return full_result

    stripped_text = decoded_text.rstrip()
    if input_ids[0, -1].item() == eos_token_id or any(
        stripped_text.endswith(marker) for marker in CHAT_END_MARKERS
    ):
        full_result["input_ids"] = input_ids
        return full_result

    seq_len = input_ids.shape[-1]
    full_result["input_ids"] = _append_sequence_value(input_ids, eos_token_id, seq_len)
    if "attention_mask" in full_result:
        full_result["attention_mask"] = _append_sequence_value(
            full_result["attention_mask"],
            1,
            seq_len,
        )
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in full_result:
            full_result[key] = _append_sequence_value(full_result[key], 0, seq_len)

    return full_result


def _add_response_labels(full_result: Dict[str, Any], tokenizer) -> Dict[str, Any]:
    input_ids = _ensure_batched_input_ids(full_result["input_ids"])
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    input_ids_flat = input_ids[0].tolist()
    decoded_text, token_offsets = _token_char_offsets(input_ids_flat, tokenizer)
    label_spans = _response_text_label_spans(decoded_text)
    _label_tokens_from_char_spans(labels, input_ids, token_offsets, label_spans)
    _label_immediate_eos_after_spans(
        labels,
        input_ids,
        token_offsets,
        label_spans,
        tokenizer,
    )

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result


def update_processor_pixels(processor, data_args):
    logger = logging.getLogger(__name__)
    model_type = getattr(data_args, "model_type", "")
    is_gemma4 = model_type == "gemma4"
    is_internvl = model_type == "internvl"
    is_molmo2 = model_type == "molmo2"

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

    def _update_optional_int_attr(component, attr_name: str, value, label: str):
        if value is None or not hasattr(component, attr_name):
            return
        setattr(component, attr_name, int(value))
        rank0_print(f"Updated {label} {attr_name} to {int(value)}")

    def _resolve_internvl_image_size():
        image_size = getattr(data_args, "internvl_image_size", None)
        if image_size is None:
            image_size = getattr(data_args, "internvl_model_image_size", None)
        if image_size is None:
            return None
        return int(image_size)

    def _resolve_molmo2_image_size():
        image_size = getattr(data_args, "molmo2_image_size", None)
        if image_size is None:
            return None
        return int(image_size)

    def _force_square_size(component, image_size: Optional[int], label: str):
        if image_size is None or not hasattr(component, "size"):
            return
        component.size = {"height": int(image_size), "width": int(image_size)}
        rank0_print(f"Updated {label}.size to InternVL square size: {component.size}")

    def _update_internvl_image_seq_length(image_size: Optional[int]):
        if not is_internvl or image_size is None:
            return

        patch_size = getattr(data_args, "internvl_patch_size", None)
        downsample_ratio = getattr(data_args, "internvl_downsample_ratio", None)
        if patch_size is None or downsample_ratio is None:
            return

        patch_size = int(patch_size)
        downsample_ratio = float(downsample_ratio)
        if image_size % patch_size != 0:
            raise ValueError(
                f"InternVL image/video size {image_size} must be divisible by "
                f"vision patch size {patch_size}. Try 448, 392, 336, or 280."
            )

        grid_size = image_size // patch_size
        pooled_grid_size = grid_size * downsample_ratio
        rounded_grid_size = int(round(pooled_grid_size))
        if abs(pooled_grid_size - rounded_grid_size) > 1e-6:
            raise ValueError(
                f"InternVL image/video size {image_size} gives patch grid "
                f"{grid_size}, which is incompatible with downsample_ratio "
                f"{downsample_ratio}. Try 448, 392, 336, or 280."
            )

        image_seq_length = rounded_grid_size * rounded_grid_size
        if hasattr(processor, "image_seq_length"):
            processor.image_seq_length = image_seq_length
            rank0_print(f"Updated InternVL image_seq_length to {image_seq_length}")

    internvl_image_size = _resolve_internvl_image_size()
    molmo2_image_size = _resolve_molmo2_image_size()
    _update_internvl_image_seq_length(internvl_image_size)

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
    rank0_print(f"Image min_patches: {getattr(ip, 'min_patches', 'N/A')}")
    rank0_print(f"Image max_patches: {getattr(ip, 'max_patches', 'N/A')}")

    if not (is_gemma4 or is_molmo2) and hasattr(ip, "min_pixels") and hasattr(ip, "max_pixels"):
        ip.min_pixels = data_args.min_pixels
        ip.max_pixels = data_args.max_pixels
        rank0_print(f"✅ Updated image_processor min_pixels to {data_args.min_pixels}")
        rank0_print(f"✅ Updated image_processor max_pixels to {data_args.max_pixels}")

    if not (is_gemma4 or is_molmo2):
        _update_edge_size(
            ip,
            shortest_edge=data_args.min_pixels,
            longest_edge=data_args.max_pixels,
            label="image_processor",
        )

    if is_internvl:
        _force_square_size(ip, internvl_image_size, "image_processor")
    if is_molmo2:
        _force_square_size(ip, molmo2_image_size, "image_processor")

    gemma4_max_soft_tokens = getattr(data_args, "gemma4_max_soft_tokens", None)
    if gemma4_max_soft_tokens is not None and hasattr(ip, "max_soft_tokens"):
        ip.max_soft_tokens = int(gemma4_max_soft_tokens)
        rank0_print(
            f"Updated image_processor max_soft_tokens to {gemma4_max_soft_tokens}"
        )

    if is_internvl:
        _update_optional_int_attr(
            ip,
            "min_patches",
            getattr(data_args, "internvl_min_patches", None),
            "image_processor",
        )
        _update_optional_int_attr(
            ip,
            "max_patches",
            getattr(data_args, "internvl_max_patches", None),
            "image_processor",
        )

    rank0_print("=== AFTER IMAGE PROCESSOR PARAMETERS ===")
    rank0_print(f"Image min_pixels: {getattr(ip, 'min_pixels', 'N/A')}")
    rank0_print(f"Image max_pixels: {getattr(ip, 'max_pixels', 'N/A')}")
    rank0_print(f"Image size: {getattr(ip, 'size', 'N/A')}")
    if hasattr(ip, "size") and isinstance(ip.size, dict):
        rank0_print(f"Image size (shortest_edge): {ip.size.get('shortest_edge', 'N/A')}")
        rank0_print(f"Image size (longest_edge):  {ip.size.get('longest_edge', 'N/A')}")
    rank0_print(f"Image max_soft_tokens: {getattr(ip, 'max_soft_tokens', 'N/A')}")
    rank0_print(f"Image min_patches: {getattr(ip, 'min_patches', 'N/A')}")
    rank0_print(f"Image max_patches: {getattr(ip, 'max_patches', 'N/A')}")

    # --- Video Processor ---
    if hasattr(processor, "video_processor") and processor.video_processor is not None:
        vp = processor.video_processor
        rank0_print("\n=== BEFORE VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video num_frames: {getattr(vp, 'num_frames', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(f"Video do_sample_frames: {getattr(vp, 'do_sample_frames', 'N/A')}")
        video_size = getattr(vp, "size", {})
        if isinstance(video_size, dict):
            rank0_print(
                f"Video size (shortest_edge): {video_size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(f"Video size (longest_edge):  {video_size.get('longest_edge', 'N/A')}")
        rank0_print(f"Video max_soft_tokens: {getattr(vp, 'max_soft_tokens', 'N/A')}")

        if not (is_gemma4 or is_molmo2) and hasattr(vp, "min_pixels") and hasattr(vp, "max_pixels"):
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

        if is_internvl:
            _update_optional_int_attr(
                vp,
                "num_frames",
                getattr(data_args, "video_max_frames", None),
                "video_processor",
            )
            if hasattr(vp, "do_sample_frames"):
                vp.do_sample_frames = True
                rank0_print("Updated video_processor do_sample_frames to True")

        if is_molmo2:
            _update_optional_int_attr(
                vp,
                "num_frames",
                getattr(data_args, "video_max_frames", None),
                "video_processor",
            )
            if hasattr(vp, "frame_sample_mode"):
                vp.frame_sample_mode = getattr(
                    data_args,
                    "molmo2_video_frame_sampling_mode",
                    "uniform_last_frame",
                )
                rank0_print(
                    f"Updated video_processor frame_sample_mode to {vp.frame_sample_mode}"
                )
            if hasattr(vp, "max_fps"):
                video_fps = getattr(data_args, "video_fps", None)
                vp.max_fps = float(video_fps) if video_fps and video_fps > 0 else None
                rank0_print(f"Updated video_processor max_fps to {vp.max_fps}")

        if hasattr(vp, "fps"):
            if is_internvl and getattr(data_args, "video_max_frames", None) is not None:
                vp.fps = None
                rank0_print(
                    "Updated video_processor fps to None because InternVL uses fixed num_frames"
                )
            else:
                vp.fps = data_args.video_fps
                rank0_print(f"✅ Updated video_processor fps to {data_args.video_fps}")

        if not (is_gemma4 or is_molmo2):
            _update_edge_size(
                vp,
                shortest_edge=data_args.video_min_pixels,
                longest_edge=data_args.video_max_pixels,
                label="video_processor",
            )

        if is_internvl:
            _force_square_size(vp, internvl_image_size, "video_processor")
        if is_molmo2:
            _force_square_size(vp, molmo2_image_size, "video_processor")

        if gemma4_max_soft_tokens is not None and hasattr(vp, "max_soft_tokens"):
            vp.max_soft_tokens = int(gemma4_max_soft_tokens)
            rank0_print(
                f"Updated video_processor max_soft_tokens to {gemma4_max_soft_tokens}"
            )

        rank0_print("=== AFTER VIDEO PROCESSOR PARAMETERS ===")
        rank0_print(f"Video min_pixels: {getattr(vp, 'min_pixels', 'N/A')}")
        rank0_print(f"Video max_pixels: {getattr(vp, 'max_pixels', 'N/A')}")
        rank0_print(f"Video num_frames: {getattr(vp, 'num_frames', 'N/A')}")
        rank0_print(f"Video min_frames: {getattr(vp, 'min_frames', 'N/A')}")
        rank0_print(f"Video max_frames: {getattr(vp, 'max_frames', 'N/A')}")
        rank0_print(f"Video fps: {getattr(vp, 'fps', 'N/A')}")
        rank0_print(f"Video do_sample_frames: {getattr(vp, 'do_sample_frames', 'N/A')}")
        rank0_print(f"Video size: {getattr(vp, 'size', 'N/A')}")
        if hasattr(vp, "size") and isinstance(vp.size, dict):
            rank0_print(
                f"Video size (shortest_edge): {vp.size.get('shortest_edge', 'N/A')}"
            )
            rank0_print(f"Video size (longest_edge):  {vp.size.get('longest_edge', 'N/A')}")
        rank0_print(f"Video max_soft_tokens: {getattr(vp, 'max_soft_tokens', 'N/A')}")

    return processor


def _build_messages(
    item: Dict[str, Any],
    base_path: Path,
    using_cot: bool = True,
    time_instruction: str = "",
    video_content_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = _normalize_media_list(item.get("images"))
    videos = _normalize_media_list(item.get("videos"))

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": _make_abs_paths(base_path, img)} for img in images
    ]
    video_content_kwargs = video_content_kwargs or {}
    video_pool = []
    for vid in videos:
        video_item = {"type": "video", "video": _make_abs_paths(base_path, vid)}
        video_item.update(video_content_kwargs)
        video_pool.append(video_item)

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
    data_args=None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))

    videos = _normalize_media_list(source.get("videos"))
    time_instruction = _build_video_time_instruction(
        base_path,
        videos,
        processor,
        data_args,
    )

    messages = _build_messages(source, base_path, using_cot, time_instruction)
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

    full_result = _add_response_labels(full_result, processor.tokenizer)
    return full_result


def preprocess_hf_chat_visual(
    sources,
    processor,
    using_cot: bool = True,
    data_args=None,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    videos = _normalize_media_list(source.get("videos"))
    time_instruction = _build_video_time_instruction(
        base_path,
        videos,
        processor,
        data_args,
    )
    messages = _build_messages(
        source,
        base_path,
        using_cot,
        time_instruction=time_instruction,
        video_content_kwargs=_build_video_content_kwargs(data_args),
    )
    template_kwargs = _build_hf_chat_template_kwargs(data_args)
    template_kwargs = _cap_hf_chat_video_num_frames(template_kwargs, base_path, videos)
    full_result = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        **template_kwargs,
    )
    if data_args is not None and getattr(data_args, "model_type", "") == "molmo2":
        full_result = _ensure_molmo2_trailing_eos(full_result, processor.tokenizer)
    full_result = _add_response_labels(full_result, processor.tokenizer)
    return full_result


def _cap_hf_chat_video_num_frames(
    template_kwargs: Dict[str, Any],
    base_path: Path,
    videos: List[str],
) -> Dict[str, Any]:
    requested_frames = _as_positive_int(template_kwargs.get("num_frames"))
    if requested_frames is None or not videos:
        return template_kwargs

    source_frame_counts = []
    for video in videos:
        _, source_frames, _ = _read_video_metadata(_make_abs_paths(base_path, video))
        if source_frames:
            source_frame_counts.append(source_frames)

    if not source_frame_counts:
        return template_kwargs

    capped_frames = min(requested_frames, min(source_frame_counts))
    if capped_frames == requested_frames:
        return template_kwargs

    template_kwargs = dict(template_kwargs)
    template_kwargs["num_frames"] = capped_frames
    return template_kwargs


def _build_video_content_kwargs(data_args) -> Dict[str, Any]:
    if data_args is None or getattr(data_args, "model_type", "") != "molmo2":
        return {}

    kwargs: Dict[str, Any] = {}
    frame_sampling_mode = getattr(
        data_args,
        "molmo2_video_frame_sampling_mode",
        None,
    )
    if frame_sampling_mode:
        kwargs["frame_sampling_mode"] = frame_sampling_mode

    video_max_frames = getattr(data_args, "video_max_frames", None)
    if video_max_frames is not None:
        kwargs["num_frames"] = int(video_max_frames)

    video_fps = getattr(data_args, "video_fps", None)
    if video_fps is not None and video_fps > 0:
        kwargs["max_fps"] = float(video_fps)

    return kwargs


def _build_hf_chat_template_kwargs(data_args) -> Dict[str, Any]:
    if data_args is None or getattr(data_args, "model_type", "") != "internvl":
        return {}

    kwargs: Dict[str, Any] = {}
    video_max_frames = getattr(data_args, "video_max_frames", None)
    if video_max_frames is not None:
        kwargs["num_frames"] = int(video_max_frames)
        kwargs["do_sample_frames"] = True

    images_kwargs: Dict[str, Any] = {}
    internvl_min_patches = getattr(data_args, "internvl_min_patches", None)
    internvl_max_patches = getattr(data_args, "internvl_max_patches", None)
    if internvl_min_patches is not None:
        images_kwargs["min_patches"] = int(internvl_min_patches)
    if internvl_max_patches is not None:
        images_kwargs["max_patches"] = int(internvl_max_patches)
    if images_kwargs:
        kwargs["images_kwargs"] = images_kwargs

    return kwargs


def preprocess_gemma4_visual(
    sources,
    processor,
    using_cot: bool = True,
    data_args=None,
) -> Dict:
    return preprocess_hf_chat_visual(sources, processor, using_cot, data_args)


def _build_minicpmv_messages_and_images(
    item: Dict[str, Any],
    base_path: Path,
    using_cot: bool,
    max_video_frames: Optional[int],
    video_group_size: int,
    time_instruction: str = "",
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
                    text_segment = seg.strip()
                    if time_instruction:
                        text_segment = f"{time_instruction}\n{text_segment}"
                        time_instruction = ""
                    content_parts.append(text_segment)

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
    videos = _normalize_media_list(source.get("videos"))
    time_instruction = _build_video_time_instruction(
        base_path,
        videos,
        processor,
        data_args,
    )
    messages, input_images, temporal_groups = _build_minicpmv_messages_and_images(
        source,
        base_path,
        getattr(data_args, "using_cot", True),
        getattr(data_args, "video_max_frames", None),
        getattr(data_args, "minicpmv_video_group_size", 6),
        time_instruction,
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
        elif data_args.model_type in {"gemma4", "internvl", "minicpmv", "molmo2"}:
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
        if self.model_type in {"gemma4", "internvl", "molmo2"}:
            data_dict = preprocess_hf_chat_visual(
                sources,
                self.processor,
                self.data_args.using_cot,
                self.data_args,
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
            return data_dict

        if self.model_type == "minicpmv":
            data_dict = preprocess_minicpmv_visual(
                sources,
                self.processor,
                self.data_args,
            )
            return data_dict

        data_dict = preprocess_qwen_visual(
            sources,
            self.processor,
            self.data_args.using_cot,
            self.data_args,
        )

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

        if any("token_type_ids" in instance for instance in instances):
            token_type_ids = []
            for instance in instances:
                if "token_type_ids" in instance:
                    token_type_id = _squeeze_leading_singleton(
                        instance["token_type_ids"]
                    )
                else:
                    token_type_id = torch.zeros(
                        instance["input_ids"].shape[-1],
                        dtype=input_ids.dtype,
                    )
                token_type_ids.append(token_type_id)
            batch["token_type_ids"] = torch.nn.utils.rnn.pad_sequence(
                token_type_ids,
                batch_first=True,
                padding_value=0,
            )[:, : self.tokenizer.model_max_length]

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
            ("image_token_pooling", 0),
            ("image_grids", 0),
            ("image_num_crops", 0),
            ("pixel_values_videos", 0),
            ("video_grid_thw", 0),
            ("video_token_pooling", 0),
            ("video_grids", 0),
            ("image_position_ids", -1),
            ("video_position_ids", -1),
            ("input_features", 0),
            ("input_features_mask", 0),
        ):
            value = _collect_tensor_field(instances, key, pad_value=pad_value)
            if value is not None:
                batch[key] = value

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

        return batch


class LazyRLDataset(Dataset):
    """Dataset for RL."""

    def __init__(self, processor, data_args):
        super().__init__()
        self.processor = processor
        self.data_args = data_args

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
                time_instruction = _build_video_time_instruction(
                    base_path,
                    videos,
                    self.processor,
                    self.data_args,
                )
                messages = _build_messages(
                    source,
                    base_path,
                    self.using_cot,
                    time_instruction,
                    video_content_kwargs=_build_video_content_kwargs(self.data_args),
                )
                template_kwargs = _build_hf_chat_template_kwargs(self.data_args)
                template_kwargs = _cap_hf_chat_video_num_frames(
                    template_kwargs,
                    base_path,
                    videos,
                )

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
                    **template_kwargs,
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
    if getattr(data_args, "model_type", "") in {"gemma4", "internvl", "minicpmv", "molmo2"} and (
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
