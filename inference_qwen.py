import argparse
import os

import torch

from inference_common import (
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_PATH,
    add_video_time_instruction,
    build_generation_kwargs,
    build_messages,
    infer_model_type,
    load_model,
    load_processor,
    prepare_processor,
    trim_repeated_response,
)


QWEN_MODEL_TYPES = {"qwen3vl", "qwen2.5vl", "qwen2vl"}


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH", "output/qwen3vl-4b-1e-bs4-ga4-merged-unique-caption")
    parser = argparse.ArgumentParser(description="Qwen-VL video inference.")
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        required=default_model_path is None,
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--model_type",
        default=os.environ.get("MODEL_TYPE", "auto"),
        choices=["auto", "qwen3vl", "qwen2.5vl", "qwen2vl"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
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
    parser.add_argument("--video_fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def resolve_qwen_model_type(args, model_path: str) -> str:
    model_type = infer_model_type(model_path) if args.model_type == "auto" else args.model_type
    if model_type not in QWEN_MODEL_TYPES:
        raise ValueError(
            f"inference_qwen.py only supports Qwen-VL checkpoints, got {model_type}. "
            "Use inference_internvl.py or inference_molmo2.py for other models."
        )
    return model_type


def main():
    args = parse_args()
    inference_model_path = args.model_path
    model_type = resolve_qwen_model_type(args, inference_model_path)
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

    messages = build_messages(args.video, args.prompt, model_type)
    add_video_time_instruction(messages, processor, args)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(model.device)

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
    print(trim_repeated_response(output_text[0]))


if __name__ == "__main__":
    main()
