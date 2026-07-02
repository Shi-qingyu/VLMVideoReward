from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import torch

from inference_common import (
    add_video_time_instruction,
    build_generation_kwargs,
    build_template_kwargs,
    build_video_content_kwargs,
    configure_internvl_processor,
    infer_model_type,
    load_model,
    load_processor,
    normalize_molmo2_messages,
    prepare_processor,
    trim_repeated_response,
)


SUPPORTED_MODEL_TYPES = {
    "qwen3vl",
    "qwen2.5vl",
    "qwen2vl",
    "internvl",
    "gemma4",
    "molmo2",
}


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH")
    parser = argparse.ArgumentParser(
        description="Run inference on one sample from physics_violation_sampled_test.json.",
    )
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        required=default_model_path is None,
    )
    parser.add_argument(
        "--dataset",
        default="data/physics_violation_sampled_test.json",
    )
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--random", action="store_true", help="Randomly choose one sample.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model_type",
        default=os.environ.get("MODEL_TYPE", "auto"),
        choices=["auto", *sorted(SUPPORTED_MODEL_TYPES)],
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument(
        "--device_map",
        default=os.environ.get("DEVICE_MAP", "auto"),
        help="Device map for model loading.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument(
        "--video_fps",
        type=float,
        default=float(os.environ.get("VIDEO_FPS", "2")),
    )
    parser.add_argument("--internvl_image_size", type=int, default=448)
    parser.add_argument("--internvl_min_patches", type=int, default=1)
    parser.add_argument("--internvl_max_patches", type=int, default=4)
    parser.add_argument(
        "--molmo2_video_frame_sampling_mode",
        default="uniform_last_frame",
    )
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def load_dataset(path: str) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    if not data:
        raise ValueError(f"{path} is empty.")
    return data


def choose_sample(data: list[dict[str, Any]], args) -> tuple[int, dict[str, Any]]:
    if args.random:
        index = random.Random(args.seed).randrange(len(data))
    else:
        index = args.index

    if index < 0 or index >= len(data):
        raise IndexError(f"--index must be in [0, {len(data) - 1}], got {index}.")
    return index, data[index]


def normalize_media_list(media: Any) -> list[str]:
    if media is None:
        return []
    if isinstance(media, str):
        return [media]
    return list(media)


def extract_human_prompt(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == "human":
            return str(turn.get("value", ""))
    raise ValueError("Sample has no human conversation turn.")


def extract_ground_truth(sample: dict[str, Any]) -> str:
    for turn in sample.get("conversations", []):
        if turn.get("from") == "gpt":
            return str(turn.get("value", ""))
    return ""


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def make_abs_media_path(media_path: str) -> str:
    path = Path(media_path)
    return str(path if path.is_absolute() else path.resolve())


def build_messages_from_sample(sample: dict[str, Any], model_type: str, args):
    prompt = extract_human_prompt(sample)
    videos = [make_abs_media_path(video) for video in normalize_media_list(sample.get("videos"))]
    images = [make_abs_media_path(image) for image in normalize_media_list(sample.get("images"))]
    video_content_kwargs = build_video_content_kwargs(model_type, args)

    content = []
    for part in re.split(r"(<image>|<video>)", prompt):
        if part == "<video>":
            if not videos:
                raise ValueError("Prompt contains <video>, but sample has no videos.")
            video_content = {"type": "video", "video": videos.pop(0)}
            video_content.update(video_content_kwargs)
            content.append(video_content)
        elif part == "<image>":
            if not images:
                raise ValueError("Prompt contains <image>, but sample has no images.")
            content.append({"type": "image", "image": images.pop(0)})
        elif part.strip():
            content.append({"type": "text", "text": part.strip()})

    if videos:
        raise ValueError(f"{len(videos)} video(s) were not consumed by <video> tags.")
    if images:
        raise ValueError(f"{len(images)} image(s) were not consumed by <image> tags.")

    messages = [{"role": "user", "content": content}]
    if model_type == "molmo2":
        messages = normalize_molmo2_messages(messages)
    return messages


def resolve_model_type(args, model_path: str) -> str:
    model_type = infer_model_type(model_path) if args.model_type == "auto" else args.model_type
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise ValueError(f"Unsupported model_type: {model_type}")
    return model_type


def move_inputs_to_model_device(inputs, model):
    device = getattr(model, "device", None)
    if device is None:
        return inputs
    return inputs.to(device)


def main():
    args = parse_args()
    data = load_dataset(args.dataset)
    sample_index, sample = choose_sample(data, args)

    inference_model_path = args.model_path
    model_type = resolve_model_type(args, inference_model_path)
    args.model_type = model_type

    model = load_model(
        inference_model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(inference_model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    if model_type == "internvl":
        configure_internvl_processor(processor, model, args)

    messages = build_messages_from_sample(sample, model_type, args)
    add_video_time_instruction(messages, processor, args)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs(model_type, args),
    )
    inputs = move_inputs_to_model_device(inputs, model)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **build_generation_kwargs(processor.tokenizer, args, model_type),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    prediction = trim_repeated_response(output_text[0])
    ground_truth = extract_ground_truth(sample)

    print(f"dataset: {args.dataset}")
    print(f"sample_index: {sample_index}")
    print(f"model_type: {model_type}")
    print(f"video: {normalize_media_list(sample.get('videos'))[0]}")
    print("\n[Prompt]")
    print(extract_human_prompt(sample))
    print("\n[Prediction]")
    print(prediction)
    print("\n[Ground Truth]")
    print(ground_truth)
    print(f"\n[pred_answer] {extract_answer(prediction)}")
    print(f"[gt_answer] {extract_answer(ground_truth)}")


if __name__ == "__main__":
    main()
