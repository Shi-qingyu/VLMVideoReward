import argparse
import os

import torch

from inference_common import (
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_PATH,
    build_generation_kwargs,
    build_messages,
    build_template_kwargs,
    configure_internvl_processor,
    infer_model_type,
    load_model,
    load_processor,
    prepare_processor,
    trim_repeated_response,
)
from src.train.checkpoint_utils import prepare_inference_model_dir


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH", "output/internvl35-4b-bs4-ga4-t-merged-unique-caption")
    parser = argparse.ArgumentParser(description="InternVL video inference.")
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
        choices=["auto", "internvl"],
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
    parser.add_argument("--internvl_image_size", type=int, default=448)
    parser.add_argument("--internvl_min_patches", type=int, default=1)
    parser.add_argument("--internvl_max_patches", type=int, default=4)
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def resolve_internvl_model_type(args, model_path: str) -> str:
    model_type = infer_model_type(model_path) if args.model_type == "auto" else args.model_type
    if model_type != "internvl":
        raise ValueError(
            f"inference_internvl.py only supports InternVL checkpoints, got {model_type}. "
            "Use inference_qwen.py or inference_molmo2.py for other models."
        )
    return model_type


def main():
    args = parse_args()
    inference_model_path = prepare_inference_model_dir(args.model_path)
    model_type = resolve_internvl_model_type(args, inference_model_path)

    model = load_model(
        inference_model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(inference_model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    configure_internvl_processor(processor, model, args)

    messages = build_messages(args.video, args.prompt, model_type)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs(model_type, args),
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
