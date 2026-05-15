import argparse
import os
from pathlib import Path

import torch

from visual_token_viz_common import (
    add_repo_root_to_path,
    first_tensor,
    get_nested_attr,
    model_device,
    move_inputs_to_device,
    print_summary,
    save_visualizations,
)


add_repo_root_to_path()

from inference_common import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_FPS,
    DEFAULT_VIDEO_PATH,
    build_messages,
    infer_model_type,
    load_model,
    load_processor,
    maybe_add_qwen_time_instruction,
    prepare_processor,
)
from src.train.checkpoint_utils import prepare_inference_model_dir  # noqa: E402


QWEN_MODEL_TYPES = {"qwen3vl", "qwen2.5vl", "qwen2vl"}


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH")
    parser = argparse.ArgumentParser(
        description=(
            "Visualize Qwen-VL visual-token features with per-frame PCA maps and "
            "adjacent-frame feature-difference heatmaps."
        )
    )
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        required=default_model_path is None,
        help="Qwen3VL/Qwen-VL checkpoint path. Can also be provided by MODEL_PATH.",
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument(
        "--model_type",
        default=os.environ.get("MODEL_TYPE", "auto"),
        choices=["auto", "qwen3vl", "qwen2.5vl", "qwen2vl"],
    )
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--video_fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION"))
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
    return parser.parse_args()


def resolve_qwen_model_type(args, model_path: str) -> str:
    model_type = infer_model_type(model_path) if args.model_type == "auto" else args.model_type
    if model_type not in QWEN_MODEL_TYPES:
        raise ValueError(f"Expected a Qwen-VL checkpoint, got model_type={model_type}.")
    return model_type


def get_visual_module(model):
    for path in ("visual", "model.visual", "model.vision_model", "vision_model"):
        module = get_nested_attr(model, path)
        if module is not None:
            return module, path
    raise ValueError("Could not find a Qwen visual module on the model.")


def get_qwen_spatial_merge_size(model, visual_module) -> int:
    for obj in (
        visual_module,
        getattr(visual_module, "config", None),
        getattr(model.config, "vision_config", None),
        getattr(model, "config", None),
    ):
        if obj is None:
            continue
        value = getattr(obj, "spatial_merge_size", None)
        if value is not None:
            return int(value)
    return 1


def reshape_qwen_visual_tokens(tokens, grid_thw, model, visual_module):
    tokens = first_tensor(tokens)
    if tokens is None:
        raise ValueError("Qwen visual module did not return a tensor.")
    if tokens.ndim == 3 and tokens.shape[0] == 1:
        tokens = tokens[0]
    if tokens.ndim != 2:
        raise ValueError(f"Expected Qwen visual tokens as [N, D], got {tuple(tokens.shape)}.")

    grid = grid_thw[0].detach().cpu().tolist()
    t, h, w = [int(x) for x in grid]
    merge = get_qwen_spatial_merge_size(model, visual_module)
    candidates = []
    if merge > 1 and h % merge == 0 and w % merge == 0:
        candidates.append((h // merge, w // merge, merge))
    candidates.append((h, w, 1))

    token_count = int(tokens.shape[0])
    for out_h, out_w, used_merge in candidates:
        expected = t * out_h * out_w
        if token_count == expected:
            print(
                f"qwen_grid_thw=({t}, {h}, {w}) spatial_merge={used_merge} "
                f"visual_grid=({t}, {out_h}, {out_w})"
            )
            return tokens.reshape(t, out_h, out_w, tokens.shape[-1])

    if token_count % t == 0:
        per_frame = token_count // t
        root = int(round(per_frame ** 0.5))
        if root * root == per_frame:
            print(
                f"qwen_grid_thw=({t}, {h}, {w}) inferred_visual_grid=({t}, {root}, {root})"
            )
            return tokens.reshape(t, root, root, tokens.shape[-1])

    raise ValueError(
        f"Cannot reshape {token_count} Qwen visual tokens from video_grid_thw={(t, h, w)}."
    )


def configure_qwen_video_processor(processor, args):
    video_processor = getattr(processor, "video_processor", None)
    if video_processor is None:
        return
    if hasattr(video_processor, "fps"):
        video_processor.fps = float(args.video_fps)


def extract_qwen_visual_features(model, processor, args, model_type: str):
    messages = build_messages(args.video, args.prompt, model_type)
    maybe_add_qwen_time_instruction(messages, processor)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = move_inputs_to_device(inputs, model_device(model))

    pixel_values = inputs.get("pixel_values_videos", None)
    grid_thw = inputs.get("video_grid_thw", None)
    if pixel_values is None:
        pixel_values = inputs.get("pixel_values", None)
        grid_thw = inputs.get("image_grid_thw", None)
    if pixel_values is None or grid_thw is None:
        raise ValueError(
            "Processor output does not contain pixel_values_videos/video_grid_thw "
            "or pixel_values/image_grid_thw."
        )

    visual_module, visual_path = get_visual_module(model)
    print(f"feature_module={visual_path}")
    with torch.inference_mode():
        try:
            visual_tokens = visual_module(pixel_values, grid_thw=grid_thw)
        except TypeError:
            visual_tokens = visual_module(pixel_values, grid_thw)
    return reshape_qwen_visual_tokens(visual_tokens, grid_thw, model, visual_module)


def main():
    args = parse_args()
    model_path = prepare_inference_model_dir(args.model_path)
    model_type = resolve_qwen_model_type(args, model_path)

    model = load_model(
        model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    configure_qwen_video_processor(processor, args)

    features = extract_qwen_visual_features(model, processor, args, model_type)
    prefix = args.prefix or f"qwen3vl_{Path(args.video).stem}"
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
