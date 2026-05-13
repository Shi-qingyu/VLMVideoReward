import argparse
import os
from pathlib import Path

from transformers import (
    AutoConfig,
    AutoImageProcessor,
    AutoModel,
    AutoModelForImageTextToText,
    AutoProcessor,
    AutoTokenizer,
    AutoVideoProcessor,
)

from src.train.checkpoint_utils import prepare_inference_model_dir


DEFAULT_MODEL_PATH = (
    "output/internvl35-4b-baseline-bs4-ga4/checkpoint-300"
)
DEFAULT_VIDEO_PATH = "data/videos/eval_0/0.mp4"
DEFAULT_PROMPT = (
    "A Black man in a short-sleeve shirt stands at a kitchen stove. He holds a box "
    "of dry pasta in one hand and pours the pasta into a pot of boiling water. "
    "Using a wooden spoon, he gently presses the pasta down to ensure it is fully "
    "submerged. The background shows kitchen counters and utensils. The camera "
    "remains steady, focusing on the man's hands and the pot."
)
QUESTION_TEMPLATE = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\n"
    "Evaluate the video according to the following dimensions.\n"
    "Video Quality: whether the video is free from major visual defects, including blur, lack of detail, "
    "poor texture, lighting issues, color distortion, flickering, and overexposure.\n"
    "Motion & Interaction: whether the subject's motion is natural, smooth, and realistic; "
    "whether interactions among subjects and/or objects are physically plausible; "
    "and whether causal relationships are correctly depicted.\n"
    "Prompt Alignment: whether the subject and object described in the prompt appear accurately, "
    "and whether the subject-object interaction described in the prompt is correctly represented.\n"
    "Prompt: {prompt} Provide your reasoning trace between think tags <think> and </think>, "
    'then output "Yes" or "No" for each dimension between <answer> and </answer>.'
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH))
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--model_type",
        default=os.environ.get("MODEL_TYPE", "auto"),
        choices=["auto", "qwen3vl", "qwen2.5vl", "qwen2vl", "internvl", "gemma4", "minicpmv"],
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--internvl_image_size", type=int, default=448)
    parser.add_argument("--internvl_min_patches", type=int, default=1)
    parser.add_argument("--internvl_max_patches", type=int, default=4)
    parser.add_argument(
        "--attn_implementation",
        default=os.environ.get("ATTN_IMPLEMENTATION"),
    )
    return parser.parse_args()


def first_int(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)


def infer_model_type(model_path: str) -> str:
    name = str(model_path).lower()
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        config_model_type = str(getattr(config, "model_type", "")).lower()
        architectures = " ".join(getattr(config, "architectures", []) or []).lower()
        haystack = f"{config_model_type} {architectures} {name}"
    except Exception:
        haystack = name

    if "internvl" in haystack:
        return "internvl"
    if "gemma" in haystack:
        return "gemma4"
    if "minicpm" in haystack:
        return "minicpmv"
    if "qwen2.5" in haystack or "qwen2_5" in haystack:
        return "qwen2.5vl"
    if "qwen2" in haystack:
        return "qwen2vl"
    return "qwen3vl"


def is_hf_internvl_checkpoint(model_path: str) -> bool:
    try:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    except Exception:
        return False
    architectures = getattr(config, "architectures", []) or []
    return (
        str(getattr(config, "model_type", "")).lower() == "internvl"
        and any("ForConditionalGeneration" in arch for arch in architectures)
    )


def token_to_id(tokenizer, token: str, required: bool = True) -> int | None:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if isinstance(token_id, list):
        token_id = token_id[0] if token_id else None
    if token_id is None or token_id == getattr(tokenizer, "unk_token_id", None):
        ids = tokenizer.encode(token, add_special_tokens=False)
        token_id = ids[0] if len(ids) == 1 else None

    if token_id is None:
        if required:
            raise ValueError(f"Could not resolve tokenizer id for InternVL token: {token}")
        return None
    return int(token_id)


def patch_internvl_tokenizer(tokenizer):
    required_token_attrs = {
        "start_image_token": "<img>",
        "end_image_token": "</img>",
        "context_image_token": "<IMG_CONTEXT>",
    }
    optional_token_attrs = {
        "image_token": "<image>",
        "video_token": "<video>",
    }
    for attr, token in {**required_token_attrs, **optional_token_attrs}.items():
        value = getattr(tokenizer, attr, None) or token
        setattr(tokenizer, attr, value)

    tokenizer.start_image_token_id = token_to_id(
        tokenizer,
        tokenizer.start_image_token,
    )
    tokenizer.end_image_token_id = token_to_id(tokenizer, tokenizer.end_image_token)
    tokenizer.context_image_token_id = token_to_id(
        tokenizer,
        tokenizer.context_image_token,
    )
    for attr in optional_token_attrs:
        token_id = token_to_id(tokenizer, getattr(tokenizer, attr), required=False)
        if token_id is not None:
            setattr(tokenizer, f"{attr}_id", token_id)
    return tokenizer


def load_processor(model_path: str, model_type: str):
    processor_kwargs = {"trust_remote_code": model_type in {"internvl", "minicpmv"}}
    if model_type != "internvl":
        return AutoProcessor.from_pretrained(model_path, **processor_kwargs)

    try:
        processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    except AttributeError as exc:
        if "start_image_token" not in str(exc):
            raise
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = patch_internvl_tokenizer(tokenizer)
        try:
            processor = AutoProcessor.from_pretrained(
                model_path,
                tokenizer=tokenizer,
                **processor_kwargs,
            )
        except AttributeError as second_exc:
            if "start_image_token" not in str(second_exc):
                raise
            from transformers.models.internvl.processing_internvl import InternVLProcessor

            config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
            image_processor = AutoImageProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            video_processor = AutoVideoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = InternVLProcessor(
                image_processor=image_processor,
                tokenizer=tokenizer,
                video_processor=video_processor,
                image_seq_length=int(getattr(config, "image_seq_length", 256)),
            )

    processor.tokenizer = patch_internvl_tokenizer(processor.tokenizer)
    return processor


def load_model(model_path: str, model_type: str, dtype: str, attn_implementation: str | None):
    model_kwargs = {
        "dtype": dtype,
        "device_map": "auto",
        "trust_remote_code": model_type in {"internvl", "minicpmv"},
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation

    if model_type == "internvl" and not is_hf_internvl_checkpoint(model_path):
        return AutoModel.from_pretrained(model_path, **model_kwargs).eval()
    return AutoModelForImageTextToText.from_pretrained(model_path, **model_kwargs).eval()


def prepare_processor(processor, model, model_type: str, model_max_length: int):
    tokenizer = processor.tokenizer
    if model_type == "internvl":
        tokenizer = patch_internvl_tokenizer(tokenizer)
        processor.tokenizer = tokenizer

    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = model_max_length

    if model_type == "internvl" and hasattr(model, "img_context_token_id"):
        model.img_context_token_id = tokenizer.convert_tokens_to_ids(
            tokenizer.context_image_token
        )


def configure_internvl_processor(processor, model, args):
    vision_config = getattr(model.config, "vision_config", None)
    model_image_size = first_int(getattr(vision_config, "image_size", None))
    image_size = int(args.internvl_image_size or model_image_size or 448)
    patch_size = int(first_int(getattr(vision_config, "patch_size", None)) or 14)
    downsample_ratio = float(getattr(model.config, "downsample_ratio", 0.5))

    if image_size % patch_size != 0:
        raise ValueError(
            f"InternVL image size {image_size} must be divisible by patch size {patch_size}."
        )

    grid_size = image_size // patch_size
    pooled_grid_size = grid_size * downsample_ratio
    rounded_grid_size = int(round(pooled_grid_size))
    if abs(pooled_grid_size - rounded_grid_size) > 1e-6:
        raise ValueError(
            f"InternVL image size {image_size} gives patch grid {grid_size}, "
            f"which is incompatible with downsample_ratio {downsample_ratio}."
        )

    if hasattr(processor, "image_seq_length"):
        processor.image_seq_length = rounded_grid_size * rounded_grid_size

    for component in (
        getattr(processor, "image_processor", None),
        getattr(processor, "video_processor", None),
    ):
        if component is not None and hasattr(component, "size"):
            component.size = {"height": image_size, "width": image_size}

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None:
        if hasattr(image_processor, "min_patches"):
            image_processor.min_patches = int(args.internvl_min_patches)
        if hasattr(image_processor, "max_patches"):
            image_processor.max_patches = int(args.internvl_max_patches)

    video_processor = getattr(processor, "video_processor", None)
    if video_processor is not None:
        if hasattr(video_processor, "num_frames"):
            video_processor.num_frames = int(args.video_max_frames)
        if hasattr(video_processor, "do_sample_frames"):
            video_processor.do_sample_frames = True
        if hasattr(video_processor, "fps"):
            video_processor.fps = None


def build_messages(video_path: str, prompt: str):
    user_input = QUESTION_TEMPLATE.format(prompt=prompt)
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": str(Path(video_path).resolve())},
                {"type": "text", "text": user_input},
            ],
        }
    ]


def get_metadata_value(metadata, key: str):
    if isinstance(metadata, dict):
        return metadata.get(key)
    return getattr(metadata, key, None)


def maybe_add_qwen_time_instruction(messages, processor):
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return

    if not hasattr(video_processor, "temporal_patch_size") or not hasattr(
        video_processor, "fps"
    ):
        return

    video_path = messages[0]["content"][0].get("video")
    videos = [video_path] if isinstance(video_path, str) else video_path
    if not videos:
        return

    vp_output = video_processor(videos=videos, return_metadata=True)
    video_grid_thw = getattr(vp_output, "video_grid_thw", None)
    video_metadata = getattr(vp_output, "video_metadata", None)
    if video_grid_thw is None or not video_metadata:
        return

    sample_fps = video_processor.fps
    temporal_patch_size = video_processor.temporal_patch_size
    total_frames = int(video_grid_thw[0][0] * temporal_patch_size)
    duration = get_metadata_value(video_metadata[0], "duration")
    if duration is None:
        return

    time_instruction = (
        f"This video is uniformly sampled at {sample_fps:.2f} fps, contains "
        f"{total_frames} frames from 0 seconds to {float(duration):.1f} seconds."
    )
    text_content = messages[0]["content"][1]
    text_content["text"] = f"{time_instruction}\n{text_content['text']}"


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


def main():
    args = parse_args()
    inference_model_path = prepare_inference_model_dir(args.model_path)
    model_type = (
        infer_model_type(inference_model_path)
        if args.model_type == "auto"
        else args.model_type
    )

    model = load_model(
        inference_model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
    )
    processor = load_processor(inference_model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    if model_type == "internvl":
        configure_internvl_processor(processor, model, args)

    messages = build_messages(args.video, args.prompt)
    if model_type.startswith("qwen"):
        maybe_add_qwen_time_instruction(messages, processor)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs(model_type, args),
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
