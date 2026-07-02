import argparse
import inspect
import os
from pathlib import Path

import torch
from transformers import AutoProcessor

from visual_token_viz_common import (
    add_repo_root_to_path,
    first_tensor,
    get_nested_attr,
    model_device,
    move_inputs_to_device,
    print_summary,
    save_visualizations,
    tokens_to_grid,
)


add_repo_root_to_path()

from inference_common import (  # noqa: E402
    load_model,
    normalize_molmo2_messages,
)


DEFAULT_MODEL_PATH = "output/molmo2-4b-baseline-bs4-ga4"
DEFAULT_VIDEO_PATH = "data/videos/eval_0/1.mp4"
DEFAULT_PROMPT = (
    "A young Black man with a beard walks through an aisle of a brightly lit toy store, surrounded by colorful shelves. "
    "He pauses in front of a shelf displaying puzzle sets, picks up a puzzle set in both hands, examines the pieces closely, "
    "and smiles at the memories of his own childhood. The camera remains steady, capturing his actions and the vibrant store setting."
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
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Molmo2 visual-token features with per-frame PCA maps and "
            "adjacent-frame feature-difference heatmaps."
        )
    )
    parser.add_argument(
        "--model_path",
        default=os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH),
        help="Molmo2 checkpoint path. Can also be provided by MODEL_PATH.",
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--raw_prompt",
        action="store_true",
        help="Use --prompt directly instead of wrapping it in the VideoReward template.",
    )
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION"))
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--video_fps", type=float, default=None)
    parser.add_argument(
        "--feature_module",
        default="auto",
        help=(
            "Dotted module path to hook, e.g. 'model.vision_backbone'. "
            "Use 'auto' to try direct feature methods first and then common vision modules."
        ),
    )
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--prefix", default=None)
    parser.add_argument(
        "--frame_indices",
        default=None,
        help="Comma/range frame list on the visual-token time axis, e.g. '0,1,2' or '0-3'.",
    )
    parser.add_argument("--pca_scope", choices=["global", "per_frame"], default="global")
    parser.add_argument("--image_scale", type=int, default=24)
    parser.add_argument("--no_normalize_diff", action="store_true")
    parser.add_argument("--save_features", action="store_true")
    parser.add_argument(
        "--print_input_keys",
        action="store_true",
        help="Print processor tensor keys/shapes before feature extraction.",
    )
    return parser.parse_args()


def build_messages(video_path: str, prompt: str, raw_prompt: bool):
    user_text = prompt if raw_prompt else QUESTION_TEMPLATE.format(prompt=prompt)
    return normalize_molmo2_messages(
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_text},
                    {"type": "video", "video": str(Path(video_path).resolve())},
                ],
            }
        ]
    )


def print_input_keys(inputs):
    for key, value in inputs.items():
        if hasattr(value, "shape"):
            dtype = getattr(value, "dtype", "")
            print(f"input[{key}] shape={tuple(value.shape)} dtype={dtype}")
        else:
            print(f"input[{key}] type={type(value).__name__}")


def infer_num_frames_from_inputs(inputs, fallback: int) -> int:
    preferred_keys = (
        "pixel_values_videos",
        "video_pixel_values",
        "videos",
        "images",
        "pixel_values",
    )
    for key in preferred_keys:
        value = inputs.get(key, None)
        if value is None or not hasattr(value, "shape"):
            continue
        shape = tuple(int(x) for x in value.shape)
        if len(shape) >= 6 and shape[0] == 1:
            return shape[1]
        if len(shape) == 5 and shape[0] == 1:
            return shape[1]
        if len(shape) == 5 and shape[0] > 1:
            return shape[0]
        if len(shape) == 4 and shape[0] > 1:
            return min(shape[0], fallback)
    return fallback


def input_aliases(inputs):
    aliases = dict(inputs)
    for source, targets in {
        "images": ("pixel_values", "image_pixel_values"),
        "pixel_values": ("images", "image_pixel_values"),
        "pixel_values_videos": ("pixel_values", "video_pixel_values"),
        "video_pixel_values": ("pixel_values_videos", "pixel_values"),
    }.items():
        if source in inputs:
            for target in targets:
                aliases.setdefault(target, inputs[source])
    return aliases


def call_with_matching_inputs(method, inputs):
    aliases = input_aliases(inputs)
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError):
        return None

    kwargs = {}
    missing_required = []
    for name, parameter in signature.parameters.items():
        if name in {"self", "args", "kwargs"}:
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            continue
        if name in aliases:
            kwargs[name] = aliases[name]
        elif parameter.default is parameter.empty:
            missing_required.append(name)

    if missing_required:
        return None
    try:
        return method(**kwargs)
    except TypeError:
        return None


def try_direct_feature_methods(model, inputs):
    method_names = (
        "get_video_features",
        "get_image_features",
        "encode_videos",
        "encode_video",
        "encode_images",
        "encode_image",
        "extract_features",
        "extract_feature",
    )
    for owner_path in ("", "model"):
        owner = model if not owner_path else get_nested_attr(model, owner_path)
        if owner is None:
            continue
        for method_name in method_names:
            method = getattr(owner, method_name, None)
            if method is None:
                continue
            print(f"trying_feature_method={owner_path + '.' if owner_path else ''}{method_name}")
            output = call_with_matching_inputs(method, inputs)
            tensor = first_tensor(output)
            if tensor is not None:
                print(f"feature_module={owner_path + '.' if owner_path else ''}{method_name}")
                return tensor
    return None


def candidate_module_paths(model):
    exact_paths = (
        "model.image_projector",
        "image_projector",
        "model.mm_projector",
        "mm_projector",
        "model.multi_modal_projector",
        "multi_modal_projector",
        "model.connector",
        "connector",
        "model.vision_backbone",
        "vision_backbone",
        "model.vision_model",
        "vision_model",
        "model.vision_tower",
        "vision_tower",
        "model.visual",
        "visual",
        "model.image_encoder",
        "image_encoder",
    )
    paths = [path for path in exact_paths if get_nested_attr(model, path) is not None]
    if paths:
        return paths

    keywords = (
        "image_projector",
        "mm_projector",
        "vision_backbone",
        "vision_model",
        "vision_tower",
        "image_encoder",
        "visual",
    )
    found = []
    for name, _module in model.named_modules():
        lowered = name.lower()
        if any(keyword in lowered for keyword in keywords):
            found.append(name)
    return sorted(found, key=lambda item: (item.count("."), item))[:24]


def forward_with_hooks(model, inputs, module_paths):
    captured = {}
    handles = []

    def make_hook(path):
        def hook(_module, _args, output):
            tensor = first_tensor(output)
            if tensor is not None and path not in captured:
                captured[path] = tensor

        return hook

    for path in module_paths:
        module = get_nested_attr(model, path)
        if module is None:
            continue
        handles.append(module.register_forward_hook(make_hook(path)))

    try:
        with torch.inference_mode():
            try:
                model(**inputs, use_cache=False, output_hidden_states=False, return_dict=True)
            except TypeError:
                model(**inputs)
    finally:
        for handle in handles:
            handle.remove()
    return captured


def choose_hooked_feature(captured, num_frames):
    best = None
    best_score = -1
    for path, tensor in captured.items():
        try:
            grid = tokens_to_grid(tensor, num_frames=num_frames, name=path)
        except Exception:
            continue
        shape = tuple(int(x) for x in grid.shape)
        name_bonus = 0
        lowered = path.lower()
        if "projector" in lowered or "connector" in lowered:
            name_bonus += 100
        if "vision" in lowered or "visual" in lowered:
            name_bonus += 50
        score = name_bonus + shape[0] * shape[1] * shape[2]
        if score > best_score:
            best = (path, grid)
            best_score = score
    if best is None:
        details = ", ".join(f"{path}:{tuple(tensor.shape)}" for path, tensor in captured.items())
        raise ValueError(f"No hooked Molmo2 feature could be reshaped. Captured: {details}")
    print(f"feature_module={best[0]} (forward hook)")
    return best[1]


def extract_molmo2_features(model, processor, args):
    messages = build_messages(args.video, args.prompt, args.raw_prompt)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    inputs = move_inputs_to_device(inputs, model_device(model))
    if args.print_input_keys:
        print_input_keys(inputs)

    num_frames = infer_num_frames_from_inputs(inputs, args.video_max_frames)
    print(f"inferred_num_frames={num_frames}")

    with torch.inference_mode():
        if args.feature_module == "auto":
            direct = try_direct_feature_methods(model, inputs)
            if direct is not None:
                return tokens_to_grid(direct, num_frames=num_frames, name="molmo2_direct_features")
            module_paths = candidate_module_paths(model)
        else:
            module_paths = [args.feature_module]

        if not module_paths:
            raise ValueError("Could not find candidate Molmo2 vision modules to hook.")
        print("hook_candidates=" + ", ".join(module_paths))
        captured = forward_with_hooks(model, inputs, module_paths)
        return choose_hooked_feature(captured, num_frames)


def configure_molmo2_video_processor(processor, args):
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    if hasattr(video_processor, "num_frames"):
        video_processor.num_frames = int(args.video_max_frames)
    if args.video_fps is not None and hasattr(video_processor, "fps"):
        video_processor.fps = float(args.video_fps)


def main():
    args = parse_args()
    model_path = args.model_path

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
    configure_molmo2_video_processor(processor, args)

    features = extract_molmo2_features(model, processor, args)
    prefix = args.prefix or f"molmo2_{Path(args.video).stem}"
    summary = save_visualizations(
        features,
        args.output_dir,
        prefix=prefix,
        video_path=args.video,
        frame_indices=args.frame_indices,
        pca_scope=args.pca_scope,
        image_scale=args.image_scale,
        normalize_for_diff=not args.no_normalize_diff,
        save_features=args.save_features,
    )
    print_summary(summary)


if __name__ == "__main__":
    main()
