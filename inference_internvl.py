import argparse
import os

from inference_common import (
    add_dataset_sample_args,
    configure_internvl_processor,
    generate_from_messages,
    get_one_dataloader_sample,
    infer_model_type,
    load_model,
    load_processor,
    print_dataloader_sample_result,
    prepare_processor,
)


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH", "output/internvl35-4b-bs4-ga4-t-merged-unique-caption")
    parser = argparse.ArgumentParser(description="InternVL video inference.")
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        required=default_model_path is None,
    )
    add_dataset_sample_args(parser)
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
    parser.add_argument(
        "--video_fps",
        type=float,
        default=float(os.environ.get("VIDEO_FPS", "2")),
    )
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
    inference_model_path = args.model_path
    model_type = resolve_internvl_model_type(args, inference_model_path)
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
    configure_internvl_processor(processor, model, args)

    if not args.dataset_use:
        raise SystemExit("--dataset_use is required for dataloader inference.")
    sample = get_one_dataloader_sample(processor, args)
    messages = sample["user"]
    prediction = generate_from_messages(model, processor, messages, model_type, args)
    print_dataloader_sample_result(sample, prediction, args, model_type)


if __name__ == "__main__":
    main()
