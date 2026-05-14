import argparse
import os

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from inference_common import (
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_PATH,
    QUESTION_TEMPLATE,
    normalize_molmo2_messages,
)
from src.train.checkpoint_utils import prepare_inference_model_dir


DEFAULT_MODEL_PATH = "output/molmo2-4b-baseline-bs4-ga4-fps2-maxf20-minf10-imgsize378-lr5e-5/checkpoint-300"


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


def build_messages(video_path: str, prompt: str, raw_prompt: bool):
    user_text = prompt if raw_prompt else QUESTION_TEMPLATE.format(prompt=prompt)
    return normalize_molmo2_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "video", "video": video_path},
                ],
            }
        ]
    )


def main():
    args = parse_args()
    model_path = prepare_inference_model_dir(args.model_path)

    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model_kwargs = {
        "trust_remote_code": True,
        "dtype": args.dtype,
        "device_map": args.device_map,
    }
    if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        **model_kwargs,
    ).eval()

    messages = build_messages(args.video, args.prompt, args.raw_prompt)
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
