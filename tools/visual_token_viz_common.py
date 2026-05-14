from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw


def add_repo_root_to_path() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    return repo_root


def get_nested_attr(obj: Any, path: str) -> Optional[Any]:
    cur = obj
    for part in path.split("."):
        if not hasattr(cur, part):
            return None
        cur = getattr(cur, part)
    return cur


def first_tensor(obj: Any) -> Optional[Any]:
    if hasattr(obj, "detach") and hasattr(obj, "shape"):
        return obj
    if hasattr(obj, "last_hidden_state"):
        tensor = first_tensor(obj.last_hidden_state)
        if tensor is not None:
            return tensor
    if hasattr(obj, "hidden_states") and obj.hidden_states:
        tensor = first_tensor(obj.hidden_states[-1])
        if tensor is not None:
            return tensor
    if isinstance(obj, dict):
        preferred = (
            "image_features",
            "video_features",
            "vision_features",
            "last_hidden_state",
            "hidden_states",
            "features",
        )
        for key in preferred:
            if key in obj:
                tensor = first_tensor(obj[key])
                if tensor is not None:
                    return tensor
        for value in obj.values():
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    if isinstance(obj, (list, tuple)):
        for value in obj:
            tensor = first_tensor(value)
            if tensor is not None:
                return tensor
    return None


def move_inputs_to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    return inputs


def model_device(model: Any) -> Any:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    try:
        return next(model.parameters()).device
    except StopIteration:
        raise ValueError("Model has no parameters; cannot infer device.")


def unwrap_feature_tensor(tensor: Any) -> Any:
    tensor = first_tensor(tensor)
    if tensor is None:
        raise ValueError("Could not find a tensor in feature output.")
    return tensor


def _shape(tensor: Any) -> tuple[int, ...]:
    return tuple(int(x) for x in tensor.shape)


def _is_square(value: int) -> bool:
    root = int(round(math.sqrt(value)))
    return root * root == value


def _sqrt_hw(num_tokens: int, name: str) -> tuple[int, int]:
    root = int(round(math.sqrt(num_tokens)))
    if root * root != num_tokens:
        raise ValueError(
            f"Cannot infer a square spatial grid for {name}: {num_tokens} tokens."
        )
    return root, root


def tokens_to_grid(
    tensor: Any,
    *,
    num_frames: Optional[int] = None,
    grid_hw: Optional[tuple[int, int]] = None,
    name: str = "features",
) -> Any:
    """Return features as [T, H, W, D].

    This accepts common vision-token shapes:
    [T, H, W, D], [T, D, H, W], [T, N, D], [1, T*N, D], or [T*N, D].
    """
    tensor = unwrap_feature_tensor(tensor)
    shape = _shape(tensor)

    if len(shape) == 5 and shape[0] == 1:
        tensor = tensor[0]
        shape = _shape(tensor)

    if len(shape) == 4:
        # Batched sequence layout [1, T, N, D].
        if shape[0] == 1 and _is_square(shape[2]):
            h, w = grid_hw or _sqrt_hw(shape[2], name)
            return tensor[0].reshape(shape[1], h, w, shape[3])
        # Already [T, H, W, D].
        if shape[-1] >= shape[1] and shape[-1] >= shape[2]:
            return tensor
        # Common vision backbone layout [T, D, H, W].
        if hasattr(tensor, "permute"):
            return tensor.permute(0, 2, 3, 1).contiguous()
        return np.transpose(tensor, (0, 2, 3, 1))

    if len(shape) == 3:
        first, tokens, dim = shape
        if first == 1 and num_frames and tokens % num_frames == 0:
            per_frame_tokens = tokens // num_frames
            h, w = grid_hw or _sqrt_hw(per_frame_tokens, name)
            return tensor.reshape(num_frames, h, w, dim)
        h, w = grid_hw or _sqrt_hw(tokens, name)
        return tensor.reshape(first, h, w, dim)

    if len(shape) == 2:
        tokens, dim = shape
        if not num_frames:
            h, w = grid_hw or _sqrt_hw(tokens, name)
            return tensor.reshape(1, h, w, dim)
        if tokens % num_frames != 0:
            raise ValueError(
                f"Cannot split {name} with shape {shape} into {num_frames} frames."
            )
        per_frame_tokens = tokens // num_frames
        h, w = grid_hw or _sqrt_hw(per_frame_tokens, name)
        return tensor.reshape(num_frames, h, w, dim)

    raise ValueError(f"Unsupported {name} shape: {shape}")


def tensor_to_numpy(tensor: Any) -> np.ndarray:
    if hasattr(tensor, "detach"):
        tensor = tensor.detach().float().cpu().numpy()
    return np.asarray(tensor, dtype=np.float32)


def parse_frame_indices(value: Optional[str], num_frames: int) -> list[int]:
    if not value:
        return list(range(num_frames))
    frames: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_s, end_s = part.split("-", 1)
            start, end = int(start_s), int(end_s)
            frames.extend(range(start, end + 1))
        else:
            frames.append(int(part))
    unique = []
    seen = set()
    for frame in frames:
        if frame < 0 or frame >= num_frames:
            raise ValueError(f"Frame index {frame} is outside [0, {num_frames}).")
        if frame not in seen:
            unique.append(frame)
            seen.add(frame)
    return unique


def l2_normalize(features: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    denom = np.linalg.norm(features, axis=-1, keepdims=True)
    return features / np.maximum(denom, eps)


def _pca_components(reference: np.ndarray, num_components: int = 3):
    x = reference.reshape(-1, reference.shape[-1]).astype(np.float32)
    x = x - x.mean(axis=0, keepdims=True)
    if x.shape[0] == 0:
        raise ValueError("Cannot run PCA on an empty feature tensor.")
    _, _, vt = np.linalg.svd(x, full_matrices=False)
    components = vt[: min(num_components, vt.shape[0])]
    if components.shape[0] < num_components:
        pad = np.zeros((num_components - components.shape[0], x.shape[1]), dtype=x.dtype)
        components = np.concatenate([components, pad], axis=0)
    mean = reference.reshape(-1, reference.shape[-1]).mean(axis=0, keepdims=True)
    return mean, components.T


def _project_pca(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    flat = features.reshape(-1, features.shape[-1]).astype(np.float32)
    projected = (flat - mean) @ components
    return projected.reshape(*features.shape[:-1], components.shape[-1])


def _scale_channels_to_uint8(values: np.ndarray, robust: bool = True) -> np.ndarray:
    values = values.astype(np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    for channel in range(values.shape[-1]):
        plane = values[..., channel]
        if robust:
            lo, hi = np.percentile(plane, [1.0, 99.0])
        else:
            lo, hi = float(plane.min()), float(plane.max())
        if hi - lo < 1e-6:
            lo, hi = float(plane.min()), float(plane.max())
        if hi - lo < 1e-6:
            out[..., channel] = 127.0
        else:
            out[..., channel] = np.clip((plane - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return out.astype(np.uint8)


def pca_rgb_maps(
    features: np.ndarray,
    frame_indices: list[int],
    *,
    scope: str = "global",
    robust: bool = True,
) -> dict[int, np.ndarray]:
    selected = features[frame_indices]
    maps: dict[int, np.ndarray] = {}

    if scope == "global":
        mean, components = _pca_components(selected)
        projected = _project_pca(selected, mean, components)
        projected_uint8 = _scale_channels_to_uint8(projected, robust=robust)
        for pos, frame_idx in enumerate(frame_indices):
            maps[frame_idx] = projected_uint8[pos]
        return maps

    if scope != "per_frame":
        raise ValueError("--pca_scope must be either 'global' or 'per_frame'.")

    for frame_idx in frame_indices:
        frame = features[frame_idx : frame_idx + 1]
        mean, components = _pca_components(frame)
        projected = _project_pca(frame, mean, components)[0]
        maps[frame_idx] = _scale_channels_to_uint8(projected, robust=robust)
    return maps


def _normalize_scalar_map(values: np.ndarray, robust: bool = True) -> np.ndarray:
    values = values.astype(np.float32)
    if robust:
        lo, hi = np.percentile(values, [1.0, 99.0])
    else:
        lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        lo, hi = float(values.min()), float(values.max())
    if hi - lo < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def apply_heatmap(values: np.ndarray, robust: bool = True) -> np.ndarray:
    x = _normalize_scalar_map(values, robust=robust)
    stops = np.asarray(
        [
            [33, 20, 73],
            [76, 29, 114],
            [122, 47, 125],
            [173, 62, 109],
            [218, 85, 82],
            [247, 139, 60],
            [253, 205, 83],
            [252, 255, 164],
        ],
        dtype=np.float32,
    )
    scaled = x * (len(stops) - 1)
    low = np.floor(scaled).astype(np.int32)
    high = np.clip(low + 1, 0, len(stops) - 1)
    alpha = scaled[..., None] - low[..., None]
    rgb = stops[low] * (1.0 - alpha) + stops[high] * alpha
    return rgb.astype(np.uint8)


def _resize_image(array: np.ndarray, scale: int) -> Image.Image:
    image = Image.fromarray(array)
    if scale > 1:
        image = image.resize((image.width * scale, image.height * scale), Image.Resampling.NEAREST)
    return image


def _draw_label(image: Image.Image, label: str) -> Image.Image:
    label_h = 20
    canvas = Image.new("RGB", (image.width, image.height + label_h), (255, 255, 255))
    canvas.paste(image.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), label, fill=(0, 0, 0))
    return canvas


def save_image_grid(
    images: Iterable[tuple[str, Image.Image]],
    output_path: Path,
    *,
    columns: int = 4,
    padding: int = 8,
) -> None:
    labeled = [_draw_label(image, label) for label, image in images]
    if not labeled:
        return
    columns = max(1, columns)
    rows = int(math.ceil(len(labeled) / columns))
    cell_w = max(image.width for image in labeled)
    cell_h = max(image.height for image in labeled)
    grid = Image.new(
        "RGB",
        (columns * cell_w + (columns + 1) * padding, rows * cell_h + (rows + 1) * padding),
        (255, 255, 255),
    )
    for idx, image in enumerate(labeled):
        row, col = divmod(idx, columns)
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        grid.paste(image.convert("RGB"), (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(output_path)


def save_visualizations(
    features: Any,
    output_dir: str | os.PathLike[str],
    *,
    prefix: str,
    frame_indices: Optional[str] = None,
    pca_scope: str = "global",
    image_scale: int = 24,
    grid_columns: int = 4,
    normalize_for_diff: bool = True,
    robust: bool = True,
    save_features: bool = False,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    arr = tensor_to_numpy(features)
    if arr.ndim != 4:
        raise ValueError(f"Expected [T, H, W, D] features, got shape {arr.shape}.")

    num_frames = arr.shape[0]
    frames = parse_frame_indices(frame_indices, num_frames)
    if not frames:
        raise ValueError("No frames selected.")

    pca_maps = pca_rgb_maps(arr, frames, scope=pca_scope, robust=robust)
    pca_grid_items = []
    for frame_idx in frames:
        image = _resize_image(pca_maps[frame_idx], image_scale)
        path = output_dir / f"{prefix}_frame{frame_idx:03d}_pca.png"
        image.save(path)
        pca_grid_items.append((f"frame {frame_idx}", image))
    save_image_grid(
        pca_grid_items,
        output_dir / f"{prefix}_pca_grid.png",
        columns=grid_columns,
    )

    diff_features = l2_normalize(arr) if normalize_for_diff else arr
    diff_grid_items = []
    diff_stats = []
    for prev_idx, next_idx in zip(frames[:-1], frames[1:]):
        diff = np.linalg.norm(diff_features[next_idx] - diff_features[prev_idx], axis=-1)
        diff_rgb = apply_heatmap(diff, robust=robust)
        image = _resize_image(diff_rgb, image_scale)
        path = output_dir / f"{prefix}_diff{prev_idx:03d}_{next_idx:03d}.png"
        image.save(path)
        diff_grid_items.append((f"{prev_idx}->{next_idx}", image))
        diff_stats.append(
            {
                "from": prev_idx,
                "to": next_idx,
                "mean": float(diff.mean()),
                "max": float(diff.max()),
                "min": float(diff.min()),
            }
        )
    save_image_grid(
        diff_grid_items,
        output_dir / f"{prefix}_diff_grid.png",
        columns=grid_columns,
    )

    if save_features:
        np.savez_compressed(output_dir / f"{prefix}_features.npz", features=arr)

    return {
        "feature_shape": list(arr.shape),
        "frames": frames,
        "pca_grid": str(output_dir / f"{prefix}_pca_grid.png"),
        "diff_grid": str(output_dir / f"{prefix}_diff_grid.png"),
        "diff_stats": diff_stats,
    }


def print_summary(summary: dict[str, Any]) -> None:
    print(f"feature_shape={summary['feature_shape']}")
    print(f"frames={summary['frames']}")
    print(f"pca_grid={summary['pca_grid']}")
    print(f"diff_grid={summary['diff_grid']}")
    if summary["diff_stats"]:
        means = [item["mean"] for item in summary["diff_stats"]]
        print(f"diff_mean_avg={float(np.mean(means)):.6f}")
