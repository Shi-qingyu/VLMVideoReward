import os
import random
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
)

from src.dataset.data_processor import (
    _build_video_time_instruction,
    make_rl_data_module,
)


REMOTE_CODE_MODEL_TYPES = {"internvl", "minicpmv", "molmo2"}


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def env_bool(name: str, default: bool) -> bool:
    return str_to_bool(os.environ[name]) if name in os.environ else default


def add_dataset_sample_args(parser):
    parser.add_argument(
        "--dataset_use",
        "--dataset_name",
        dest="dataset_use",
        default=os.environ.get("DATASET_USE") or os.environ.get("DATASET"),
        help="Dataset name registered in src/dataset/__init__.py.",
    )
    parser.add_argument(
        "--sample_index",
        "--index",
        dest="sample_index",
        type=int,
        default=int(os.environ.get("SAMPLE_INDEX", "0")),
    )
    parser.add_argument(
        "--random_sample",
        "--random",
        dest="random_sample",
        action="store_true",
    )
    parser.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "42")))
    parser.add_argument(
        "--using_cot",
        nargs="?",
        const=True,
        default=env_bool("USING_COT", True),
        type=str_to_bool,
    )
    parser.add_argument("--no_using_cot", action="store_false", dest="using_cot")
    return parser


def infer_model_type(model_path: str) -> str:
    name = str(model_path).lower()
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        arch = " ".join(getattr(config, "architectures", []) or []).lower()
        haystack = f"{getattr(config, 'model_type', '')} {arch} {name}".lower()
    except Exception:
        haystack = name

    if "internvl" in haystack:
        return "internvl"
    if "molmo" in haystack:
        return "molmo2"
    if "gemma" in haystack:
        return "gemma4"
    if "minicpm" in haystack:
        return "minicpmv"
    if "qwen2.5" in haystack or "qwen2_5" in haystack:
        return "qwen2.5vl"
    if "qwen2" in haystack:
        return "qwen2vl"
    return "qwen3vl"


def load_processor(model_path: str, model_type: str):
    processor = AutoProcessor.from_pretrained(
        model_path,
        trust_remote_code=model_type in REMOTE_CODE_MODEL_TYPES,
    )
    if model_type == "molmo2" and not hasattr(processor, "audio_tokenizer"):
        processor.audio_tokenizer = None
    return processor


def load_model(
    model_path: str,
    model_type: str,
    dtype: str,
    attn_implementation: str | None,
    device_map: str | None = "auto",
):
    model_kwargs = {
        "dtype": dtype,
        "trust_remote_code": model_type in REMOTE_CODE_MODEL_TYPES,
    }
    if device_map not in {None, "", "none"}:
        model_kwargs["device_map"] = device_map
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    model_cls = AutoModelForImageTextToText
    if (
        model_type == "internvl"
        and not str(model_path).rstrip("/").lower().endswith("-hf")
    ):
        model_cls = AutoModel

    return model_cls.from_pretrained(model_path, **model_kwargs).eval()


def prepare_processor(processor, model, model_type: str, model_max_length: int):
    tokenizer = processor.tokenizer
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_max_length

    if model_type == "internvl":
        tokenizer.start_image_token = getattr(tokenizer, "start_image_token", "<img>")
        tokenizer.end_image_token = getattr(tokenizer, "end_image_token", "</img>")
        tokenizer.context_image_token = getattr(
            tokenizer,
            "context_image_token",
            "<IMG_CONTEXT>",
        )
        if hasattr(model, "img_context_token_id"):
            model.img_context_token_id = tokenizer.convert_tokens_to_ids(
                tokenizer.context_image_token,
            )


def configure_internvl_processor(processor, model, args):
    image_size = int(args.internvl_image_size)
    vision_config = getattr(model.config, "vision_config", None)
    patch_size = getattr(vision_config, "patch_size", 14)
    patch_size = patch_size[0] if isinstance(patch_size, (list, tuple)) else patch_size
    downsample_ratio = float(getattr(model.config, "downsample_ratio", 0.5))

    if hasattr(processor, "image_seq_length"):
        pooled_grid = int(round((image_size // int(patch_size)) * downsample_ratio))
        processor.image_seq_length = pooled_grid * pooled_grid

    for component in (processor.image_processor, processor.video_processor):
        if hasattr(component, "size"):
            component.size = {"height": image_size, "width": image_size}

    image_processor = processor.image_processor
    if hasattr(image_processor, "min_patches"):
        image_processor.min_patches = int(args.internvl_min_patches)
    if hasattr(image_processor, "max_patches"):
        image_processor.max_patches = int(args.internvl_max_patches)

    video_processor = processor.video_processor
    if hasattr(video_processor, "num_frames"):
        video_processor.num_frames = int(args.video_max_frames)
    if hasattr(video_processor, "do_sample_frames"):
        video_processor.do_sample_frames = True
    if hasattr(video_processor, "fps"):
        video_processor.fps = None


def build_video_content_kwargs(model_type: str, args) -> dict:
    if model_type != "molmo2":
        return {}

    kwargs = {
        "num_frames": int(args.video_max_frames),
        "frame_sampling_mode": args.molmo2_video_frame_sampling_mode,
    }
    if args.video_fps > 0:
        kwargs["max_fps"] = float(args.video_fps)
    return kwargs


def normalize_molmo2_messages(messages):
    normalized = []
    for message in messages:
        content = message.get("content", [])
        if message.get("role") == "user" and isinstance(content, list):
            text_parts = [part for part in content if part.get("type") == "text"]
            media_parts = [part for part in content if part.get("type") != "text"]
            message = {**message, "content": text_parts + media_parts}
        normalized.append(message)
    return normalized


def add_video_time_instruction(messages, processor, args=None):
    content = messages[0]["content"]
    video_paths = [part["video"] for part in content if part.get("type") == "video"]
    if not video_paths:
        return

    time_instruction = _build_video_time_instruction(
        Path(""),
        video_paths,
        processor,
        args,
    )
    if not time_instruction:
        return

    text_part = next(part for part in content if part.get("type") == "text")
    text_part["text"] = f"{time_instruction}\n{text_part['text']}"


def build_template_kwargs(model_type: str, args):
    if model_type != "internvl":
        return {}
    return {
        "num_frames": int(args.video_max_frames),
        "do_sample_frames": True,
        "images_kwargs": {
            "min_patches": int(args.internvl_min_patches),
            "max_patches": int(args.internvl_max_patches),
        },
    }


def collect_eos_token_ids(tokenizer) -> list[int]:
    eos_ids = []
    eos_token_id = tokenizer.eos_token_id
    if isinstance(eos_token_id, (list, tuple)):
        eos_ids.extend(eos_token_id)
    elif eos_token_id is not None:
        eos_ids.append(eos_token_id)

    for token in ("<|im_end|>", "</s>", tokenizer.eos_token):
        if token is None:
            continue
        token_id = tokenizer.convert_tokens_to_ids(token)
        if isinstance(token_id, int) and token_id != tokenizer.unk_token_id:
            eos_ids.append(token_id)
    return sorted(set(eos_ids))


def build_generation_kwargs(tokenizer, args, model_type: str = ""):
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.temperature > 0,
        "pad_token_id": tokenizer.pad_token_id,
    }

    eos_token_ids = collect_eos_token_ids(tokenizer)
    if eos_token_ids:
        generation_kwargs["eos_token_id"] = eos_token_ids
    if args.temperature > 0:
        generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    if model_type != "molmo2":
        generation_kwargs["repetition_penalty"] = args.repetition_penalty
        if args.no_repeat_ngram_size > 0:
            generation_kwargs["no_repeat_ngram_size"] = args.no_repeat_ngram_size
    return generation_kwargs


def get_one_dataloader_sample(processor, args):
    data_module = make_rl_data_module(processor=processor, data_args=args)
    dataset = data_module["train_dataset"]
    if args.random_sample:
        sample_index = random.Random(args.seed).randrange(len(dataset))
    else:
        sample_index = int(args.sample_index)

    dataloader = DataLoader(
        Subset(dataset, [sample_index]),
        collate_fn=data_module["data_collator"],
        batch_size=1,
    )
    batch = next(iter(dataloader))
    return {
        "sample_index": sample_index,
        "dataset_size": len(dataset),
        "user": batch["user"][0],
        "gt": batch["gt"][0],
    }


def move_inputs_to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
    }


def generate_from_messages(model, processor, messages, model_type: str, args) -> str:
    if model_type == "molmo2":
        messages = normalize_molmo2_messages(messages)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs(model_type, args),
    )
    inputs = move_inputs_to_device(inputs, model.device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            **build_generation_kwargs(processor.tokenizer, args, model_type),
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return trim_repeated_response(output_text[0])


def extract_first_media(messages, media_type: str):
    for message in messages:
        for part in message["content"]:
            if part.get("type") == media_type:
                return part[media_type]
    return None


def extract_text_from_messages(messages) -> str:
    return "\n".join(
        part["text"]
        for message in messages
        for part in message["content"]
        if part.get("type") == "text"
    )


def extract_answer(text: str) -> str:
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else ""


def print_dataloader_sample_result(sample, prediction: str, args, model_type: str):
    ground_truth = extract_text_from_messages(sample["gt"])

    print(f"model_path: {args.model_path}")
    print(f"dataset_name: {args.dataset_use}")
    print(f"sample_index: {sample['sample_index']} / {sample['dataset_size']}")
    print(f"model_type: {model_type}")
    for media_type in ("video", "image"):
        media_path = extract_first_media(sample["user"], media_type)
        if media_path:
            print(f"{media_type}: {media_path}")

    print("\n[Prompt]")
    print(extract_text_from_messages(sample["user"]))
    print("\n[Prediction]")
    print(prediction)
    print("\n[Ground Truth]")
    print(ground_truth)

    pred_answer = extract_answer(prediction)
    gt_answer = extract_answer(ground_truth)
    if pred_answer or gt_answer:
        print(f"\n[pred_answer] {pred_answer}")
        print(f"[gt_answer] {gt_answer}")


def trim_repeated_response(text: str) -> str:
    for marker in ("</answer>", "<|im_end|>", "</s>"):
        index = text.find(marker)
        if index >= 0:
            return text[: index + len(marker)].strip()
    return text.strip()
