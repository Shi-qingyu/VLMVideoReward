import argparse
import os
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference_common import load_model  # noqa: E402


DEFAULT_MODEL_PATH = "output/molmo2-4b-baseline-bs4-ga4"
DEFAULT_VIDEO_PATH = "data/videos/eval_0/0.mp4"
DEFAULT_PROMPT = (
    "A young Black man with a beard walks through an aisle of a brightly lit toy store, surrounded by colorful shelves. "
    "He pauses in front of a shelf displaying puzzle sets, picks up a puzzle set in both hands, examines the pieces closely, "
    "and smiles at the memories of his own childhood. The camera remains steady, capturing his actions and the vibrant store setting."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Molmo2 video inference.")
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    return parser.parse_args()


def move_inputs_to_device(inputs, device):
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def main():
    args = parse_args()

    processor = AutoProcessor.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        dtype=args.dtype,
        device_map=args.device_map,
    )
    model = load_model(
        args.model_path,
        "molmo2",
        dtype=args.dtype,
        attn_implementation=None,
        device_map=args.device_map,
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": args.prompt},
                {"type": "video", "video": args.video},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = move_inputs_to_device(inputs, model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
        )

    generated_tokens = generated_ids[0, inputs["input_ids"].size(1) :]
    generated_text = processor.tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True,
    )
    print(generated_text)


if __name__ == "__main__":
    main()
