import argparse
import os

import torch
from transformers import AutoProcessor

from inference_common import (
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_PATH,
    QUESTION_TEMPLATE,
    add_video_time_instruction,
    build_video_content_kwargs,
    load_model,
    normalize_molmo2_messages,
)


DEFAULT_MODEL_PATH = "output/molmo2-4b-baseline-bs4-ga4-t-merged-unique/checkpoint-200"


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Molmo2 video inference.")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Use --prompt directly instead of wrapping it in the VideoReward judging template.",
    )
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--video_fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument(
        "--molmo2_video_frame_sampling_mode",
        default="uniform_last_frame",
    )
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def move_inputs_to_device(inputs, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def build_messages(video_path: str, prompt: str, raw_prompt: bool, args):
    user_text = prompt if raw_prompt else QUESTION_TEMPLATE.format(prompt=prompt)
    video_content = {"type": "video", "video": video_path}
    video_content.update(build_video_content_kwargs("molmo2", args))
    return normalize_molmo2_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    video_content,
                ],
            }
        ]
    )


def main():
    args = parse_args()
    model_path = args.model_path
    args.model_type = "molmo2"

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model = load_model(
        model_path,
        "molmo2",
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )

    messages = build_messages(args.video, args.prompt, args.raw_prompt, args)
    add_video_time_instruction(messages, processor, args)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = move_inputs_to_device(inputs, model.device)

    with torch.inference_mode():
        generation_kwargs = {
            "max_new_tokens": args.max_new_tokens,
            "do_sample": args.temperature > 0,
        }
        if args.temperature > 0:
            generation_kwargs["temperature"] = args.temperature
            generation_kwargs["top_p"] = args.top_p
        generated_ids = model.generate(**inputs, **generation_kwargs)

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
