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
    tokens_to_grid,
)


add_repo_root_to_path()

from inference_common import (  # noqa: E402
    DEFAULT_PROMPT,
    DEFAULT_VIDEO_PATH,
    build_messages,
    build_template_kwargs,
    configure_internvl_processor,
    infer_model_type,
    load_model,
    load_processor,
    prepare_processor,
)


def parse_args():
    default_model_path = os.environ.get("MODEL_PATH")
    parser = argparse.ArgumentParser(
        description=(
            "Visualize InternVL visual-token features with per-frame PCA maps and "
            "adjacent-frame feature-difference heatmaps."
        )
    )
    parser.add_argument(
        "--model_path",
        default=default_model_path,
        required=default_model_path is None,
        help="InternVL checkpoint path. Can also be provided by MODEL_PATH.",
    )
    parser.add_argument("--video", default=os.environ.get("VIDEO", DEFAULT_VIDEO_PATH))
    parser.add_argument("--prompt", default=os.environ.get("PROMPT", DEFAULT_PROMPT))
    parser.add_argument("--model_type", default=os.environ.get("MODEL_TYPE", "auto"))
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default=os.environ.get("DTYPE", "auto"))
    parser.add_argument("--device_map", default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--internvl_image_size", type=int, default=448)
    parser.add_argument(
        "--internvl_min_patches",
        type=int,
        default=1,
        help="Keep this at 1 for frame-aligned visualizations.",
    )
    parser.add_argument(
        "--internvl_max_patches",
        type=int,
        default=1,
        help="Keep this at 1 for frame-aligned visualizations.",
    )
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


def resolve_internvl_model_type(args, model_path: str) -> str:
    model_type = infer_model_type(model_path) if args.model_type == "auto" else args.model_type
    if model_type != "internvl":
        raise ValueError(f"Expected an InternVL checkpoint, got model_type={model_type}.")
    return model_type


def get_pixel_values(inputs):
    for key in ("pixel_values", "pixel_values_videos", "images"):
        value = inputs.get(key, None)
        if value is not None and hasattr(value, "shape") and value.ndim >= 4:
            return value, key
    raise ValueError("Could not find InternVL pixel values in processor output.")


def run_vision_fallback(model, pixel_values):
    for path in (
        "vision_model",
        "model.vision_model",
        "vision_tower",
        "model.vision_tower",
        "visual",
        "model.visual",
    ):
        module = get_nested_attr(model, path)
        if module is None:
            continue
        print(f"feature_module={path} (raw vision fallback)")
        try:
            output = module(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
        except TypeError:
            output = module(pixel_values)
        tensor = first_tensor(output)
        if tensor is not None:
            if tensor.ndim == 3 and tensor.shape[1] > 1:
                no_cls = tensor[:, 1:, :]
                per_image_tokens = int(no_cls.shape[1])
                root = int(round(per_image_tokens ** 0.5))
                if root * root == per_image_tokens:
                    return no_cls
            return tensor
    raise ValueError("Could not extract InternVL vision features.")


def extract_internvl_features(model, processor, args):
    messages = build_messages(args.video, args.prompt, "internvl")
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs("internvl", args),
    )
    inputs = move_inputs_to_device(inputs, model_device(model))
    pixel_values, pixel_key = get_pixel_values(inputs)
    print(f"pixel_key={pixel_key} pixel_shape={tuple(pixel_values.shape)}")

    if args.internvl_max_patches != 1:
        print(
            "warning: internvl_max_patches > 1 may make the first axis represent "
            "frame tiles rather than frames."
        )

    with torch.inference_mode():
        if hasattr(model, "extract_feature"):
            print("feature_module=extract_feature")
            features = model.extract_feature(pixel_values)
        elif hasattr(model, "get_image_features"):
            print("feature_module=get_image_features")
            try:
                features = model.get_image_features(pixel_values=pixel_values)
            except TypeError:
                features = model.get_image_features(pixel_values)
        else:
            features = run_vision_fallback(model, pixel_values)

    tensor = first_tensor(features)
    if tensor is None:
        raise ValueError("InternVL feature extractor did not return a tensor.")

    num_frames = int(pixel_values.shape[0]) if pixel_values.ndim == 4 else args.video_max_frames
    return tokens_to_grid(tensor, num_frames=num_frames, name="internvl_features")


def main():
    args = parse_args()
    model_path = args.model_path
    model_type = resolve_internvl_model_type(args, model_path)

    model = load_model(
        model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    configure_internvl_processor(processor, model, args)

    features = extract_internvl_features(model, processor, args)
    prefix = args.prefix or f"internvl_{Path(args.video).stem}"
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
