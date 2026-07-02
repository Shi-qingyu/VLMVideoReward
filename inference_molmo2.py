import argparse
import os

from inference_common import (
    add_dataset_sample_args,
    generate_from_messages,
    get_one_dataloader_sample,
    load_model,
    load_processor,
    prepare_processor,
    print_dataloader_sample_result,
)


DEFAULT_MODEL_PATH = "output/molmo2-4b-baseline-bs4-ga4-t-merged-unique/checkpoint-200"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Molmo2 video inference.")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
    )
    add_dataset_sample_args(parser)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument(
        "--video_fps",
        type=float,
        default=float(os.environ.get("VIDEO_FPS", "2")),
    )
    parser.add_argument(
        "--molmo2_video_frame_sampling_mode",
        default="uniform_last_frame",
    )
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = args.model_path
    args.model_type = "molmo2"

    model = load_model(
        model_path,
        "molmo2",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(model_path, "molmo2")
    prepare_processor(processor, model, "molmo2", args.model_max_length)

    if not args.dataset_use:
        raise SystemExit("--dataset_use is required for dataloader inference.")
    sample = get_one_dataloader_sample(processor, args)
    messages = sample["user"]
    prediction = generate_from_messages(model, processor, messages, "molmo2", args)
    print_dataloader_sample_result(sample, prediction, args, "molmo2")


if __name__ == "__main__":
    main()
