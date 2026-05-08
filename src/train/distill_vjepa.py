import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _ensure_vjepa2_path() -> Path:
    vjepa2_root = _project_root() / "vjepa2"
    if not vjepa2_root.exists():
        raise FileNotFoundError(
            f"Missing V-JEPA code directory: {vjepa2_root}. Expected a local vjepa2/ checkout."
        )
    vjepa2_root_str = str(vjepa2_root)
    if vjepa2_root_str not in sys.path:
        sys.path.insert(0, vjepa2_root_str)
    return vjepa2_root


def _resolve_checkpoint_path(checkpoint_path: str) -> Path:
    candidate = Path(checkpoint_path)
    if candidate.is_absolute():
        path = candidate
    else:
        path = (_project_root() / candidate).resolve()
        if not path.exists():
            path = (Path.cwd() / candidate).resolve()
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find V-JEPA checkpoint: {checkpoint_path}. "
            f"Checked {(_project_root() / candidate).resolve()} and {(Path.cwd() / candidate).resolve()}."
        )
    return path


def _clean_state_dict(state_dict: dict[str, Any]) -> dict[str, Any]:
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "")
        key = key.replace("backbone.", "")
        cleaned[key] = value
    return cleaned


def _unwrap_teacher_state_dict(checkpoint: dict[str, Any]) -> dict[str, Any]:
    for key in ("ema_encoder", "target_encoder", "encoder", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return _clean_state_dict(value)
    return _clean_state_dict(checkpoint)


def _load_vjepa2_builder(teacher_arch: str):
    _ensure_vjepa2_path()
    from src.hub import backbones as vjepa_backbones

    if not hasattr(vjepa_backbones, teacher_arch):
        raise ValueError(
            f"Unsupported V-JEPA teacher arch: {teacher_arch}. "
            f"Expected one of the builders exposed in vjepa2/src/hub/backbones.py."
        )
    return getattr(vjepa_backbones, teacher_arch)


def build_vjepa2_teacher(teacher_arch: str, checkpoint_path: str) -> nn.Module:
    builder = _load_vjepa2_builder(teacher_arch)
    encoder, _ = builder(pretrained=False)
    ckpt_path = _resolve_checkpoint_path(checkpoint_path)
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = _unwrap_teacher_state_dict(checkpoint)
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("V-JEPA teacher missing keys: %s", missing)
    if unexpected:
        logger.warning("V-JEPA teacher unexpected keys: %s", unexpected)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    return encoder


def _flatten_path_batches(path_batches: Optional[Sequence[Sequence[str]]]) -> list[str]:
    if not path_batches:
        return []
    return [path for batch in path_batches for path in batch]


def _find_visual_module(model: nn.Module) -> nn.Module:
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        visual = getattr(current, "visual", None)
        if isinstance(visual, nn.Module):
            return visual
        queue.extend(
            getattr(current, attr, None)
            for attr in ("module", "model", "base_model")
        )
    raise AttributeError("Could not locate Qwen visual module on the provided model.")


def _extract_primary_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (list, tuple)):
        for item in output:
            if isinstance(item, torch.Tensor):
                return item
    if isinstance(output, dict):
        for key in ("last_hidden_state", "image_embeds", "video_embeds", "hidden_states"):
            value = output.get(key)
            if isinstance(value, torch.Tensor):
                return value
            if isinstance(value, (list, tuple)) and value and isinstance(value[-1], torch.Tensor):
                return value[-1]
    raise TypeError(f"Unsupported visual output type: {type(output)}")


def select_student_features(output: Any, feature_source: str) -> torch.Tensor:
    if feature_source == "visual":
        tensor = _extract_primary_tensor(output)
    elif feature_source == "deepstack_last":
        if not isinstance(output, (list, tuple)) or len(output) < 2:
            raise ValueError(
                "Requested deepstack distillation but Qwen visual output does not expose deepstack features."
            )
        deepstack = output[1]
        if not isinstance(deepstack, (list, tuple)) or not deepstack:
            raise ValueError("Qwen deepstack feature list is empty.")
        tensor = deepstack[-1]
    else:
        raise ValueError(f"Unsupported distill_feature_source: {feature_source}")

    if tensor.ndim == 3 and tensor.size(0) == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D visual tokens, got shape {tuple(tensor.shape)}")
    return tensor


def infer_student_token_shape(
    grid_item: torch.Tensor,
    merge_size: int,
) -> Tuple[int, int, int]:
    t, h, w = [int(x) for x in grid_item.detach().cpu().tolist()]
    merged_h = max(h // max(merge_size, 1), 1)
    merged_w = max(w // max(merge_size, 1), 1)
    return t, merged_h, merged_w


def split_visual_tokens_with_shapes(
    tokens: Optional[torch.Tensor],
    grid_thw: Optional[torch.Tensor],
    merge_size: int,
) -> list[Tuple[torch.Tensor, Tuple[int, int, int]]]:
    if tokens is None or grid_thw is None:
        return []

    shapes = [infer_student_token_shape(item, merge_size) for item in grid_thw]
    counts = [int(t * h * w) for t, h, w in shapes]

    if not counts:
        return []
    if sum(counts) != int(tokens.shape[0]):
        raise ValueError(
            "Visual token count mismatch while splitting media items: "
            f"sum(counts)={sum(counts)} vs tokens={tokens.shape[0]}"
        )
    chunks = list(tokens.split(counts, dim=0))
    return list(zip(chunks, shapes))


def reshape_tokens_to_grid(
    tokens: torch.Tensor,
    shape: Tuple[int, int, int],
) -> torch.Tensor:
    t, h, w = shape
    expected = int(t * h * w)
    if int(tokens.shape[0]) != expected:
        raise ValueError(
            f"Cannot reshape tokens of shape {tuple(tokens.shape)} into grid {shape}."
        )
    return tokens.view(t, h, w, tokens.shape[-1])


def infer_teacher_grid_shape(
    tokens: torch.Tensor,
    student_shape: Tuple[int, int, int],
    modality: str,
) -> Tuple[int, int, int]:
    if modality == "image":
        teacher_t = 1
    elif modality == "video":
        teacher_t = max(int(student_shape[0]), 1)
    else:
        raise ValueError(f"Unsupported modality: {modality}")

    if int(tokens.shape[0]) % teacher_t != 0:
        raise ValueError(
            f"Teacher token count {tokens.shape[0]} is not divisible by temporal size {teacher_t}."
        )

    spatial_tokens = int(tokens.shape[0]) // teacher_t
    side = int(round(math.sqrt(spatial_tokens)))
    if side * side == spatial_tokens:
        return teacher_t, side, side
    return teacher_t, spatial_tokens, 1


def align_teacher_tokens_to_student_shape(
    teacher_tokens: torch.Tensor,
    student_shape: Tuple[int, int, int],
    modality: str,
) -> torch.Tensor:
    teacher_shape = infer_teacher_grid_shape(
        teacher_tokens,
        student_shape=student_shape,
        modality=modality,
    )
    teacher_grid = reshape_tokens_to_grid(teacher_tokens, teacher_shape)
    teacher_grid = teacher_grid.permute(3, 0, 1, 2).unsqueeze(0).contiguous()

    if teacher_shape != student_shape:
        teacher_grid = F.adaptive_avg_pool3d(teacher_grid, output_size=student_shape)

    aligned = teacher_grid.squeeze(0).permute(1, 2, 3, 0).contiguous()
    return aligned.view(-1, aligned.shape[-1])


class VJepa2TeacherEncoder:
    def __init__(
        self,
        teacher_arch: str,
        checkpoint_path: str,
        image_size: int = 384,
        num_video_frames: int = 16,
    ):
        self.teacher = build_vjepa2_teacher(teacher_arch, checkpoint_path)
        self.teacher_dim = int(getattr(self.teacher, "embed_dim"))
        self.teacher_tubelet_size = int(getattr(self.teacher, "tubelet_size", 2))
        self.image_size = int(image_size)
        self.short_side_size = int(round(self.image_size * 256 / 224))
        self.num_video_frames = int(num_video_frames)
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None

    def prepare(self, device: torch.device, dtype: torch.dtype) -> None:
        if self._device != device or self._dtype != dtype:
            self.teacher.to(device=device, dtype=dtype)
            self.teacher.eval()
            self._device = device
            self._dtype = dtype

    def _resize_short_side(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        if min(height, width) == self.short_side_size:
            return frames

        if height <= width:
            new_height = self.short_side_size
            new_width = int(round(width * self.short_side_size / height))
        else:
            new_width = self.short_side_size
            new_height = int(round(height * self.short_side_size / width))

        return F.interpolate(
            frames,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )

    def _center_crop(self, frames: torch.Tensor) -> torch.Tensor:
        _, _, height, width = frames.shape
        crop_h = min(self.image_size, height)
        crop_w = min(self.image_size, width)
        top = max((height - crop_h) // 2, 0)
        left = max((width - crop_w) // 2, 0)
        frames = frames[:, :, top : top + crop_h, left : left + crop_w]
        if crop_h != self.image_size or crop_w != self.image_size:
            frames = F.interpolate(
                frames,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return frames

    def _preprocess_frames(self, frames_thwc: torch.Tensor) -> torch.Tensor:
        if frames_thwc.ndim != 4:
            raise ValueError(
                f"Expected frames with shape [T, H, W, C], got {tuple(frames_thwc.shape)}"
            )

        frames = frames_thwc.to(torch.float32)
        if frames.max() > 1.0:
            frames = frames / 255.0
        frames = frames.permute(0, 3, 1, 2).contiguous()
        frames = self._resize_short_side(frames)
        frames = self._center_crop(frames)

        mean = torch.tensor(IMAGENET_MEAN, dtype=frames.dtype).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, dtype=frames.dtype).view(1, 3, 1, 1)
        frames = (frames - mean) / std
        return frames.permute(1, 0, 2, 3).contiguous()

    def _sample_frame_indices(self, total_frames: int, desired_frames: int) -> torch.Tensor:
        if total_frames <= 0:
            raise ValueError(f"Video contains no frames, got total_frames={total_frames}")
        desired_frames = max(int(desired_frames), self.teacher_tubelet_size)
        if desired_frames % self.teacher_tubelet_size != 0:
            desired_frames = (
                (desired_frames + self.teacher_tubelet_size - 1)
                // self.teacher_tubelet_size
            ) * self.teacher_tubelet_size
        if desired_frames == 1 or total_frames == 1:
            return torch.zeros(desired_frames, dtype=torch.long)
        return torch.linspace(
            0, total_frames - 1, steps=desired_frames, dtype=torch.float32
        ).round().to(torch.long)

    def _load_image_clip(self, image_path: str) -> torch.Tensor:
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            frames = torch.from_numpy(np.array(image)).unsqueeze(0)
        return self._preprocess_frames(frames).unsqueeze(0)

    def _load_video_clip(
        self,
        video_path: str,
        target_temporal_tokens: Optional[int] = None,
    ) -> torch.Tensor:
        frames = None
        video_error: Optional[Exception] = None
        if target_temporal_tokens is None:
            desired_frames = self.num_video_frames
        else:
            desired_frames = int(target_temporal_tokens) * self.teacher_tubelet_size

        try:
            from decord import VideoReader, cpu

            reader = VideoReader(video_path, ctx=cpu(0))
            indices = self._sample_frame_indices(len(reader), desired_frames)
            frames = torch.from_numpy(reader.get_batch(indices.tolist()).asnumpy())
        except Exception as exc:
            video_error = exc

        if frames is None:
            try:
                from torchvision.io import read_video

                video, _, _ = read_video(video_path, pts_unit="sec")
                indices = self._sample_frame_indices(int(video.shape[0]), desired_frames)
                frames = video.index_select(0, indices)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load video {video_path} with decord and torchvision."
                ) from (video_error or exc)

        if int(frames.shape[0]) == 1:
            frames = torch.cat([frames, frames[-1:].clone()], dim=0)
        elif int(frames.shape[0]) % 2 != 0:
            frames = torch.cat([frames, frames[-1:].clone()], dim=0)

        return self._preprocess_frames(frames).unsqueeze(0)

    @torch.no_grad()
    def encode_images(
        self,
        image_paths: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if not image_paths:
            return []
        self.prepare(device, dtype)
        features = []
        for image_path in image_paths:
            clip = self._load_image_clip(image_path).to(device=device, dtype=dtype)
            encoded = self.teacher(clip)
            features.append(encoded.squeeze(0).float())
        return features

    @torch.no_grad()
    def encode_videos(
        self,
        video_paths: Sequence[str],
        device: torch.device,
        dtype: torch.dtype,
        target_temporal_tokens: Optional[Sequence[int]] = None,
    ) -> list[torch.Tensor]:
        if not video_paths:
            return []
        if target_temporal_tokens is not None and len(video_paths) != len(target_temporal_tokens):
            raise ValueError(
                "target_temporal_tokens length must match video_paths length: "
                f"{len(target_temporal_tokens)} vs {len(video_paths)}"
            )
        self.prepare(device, dtype)
        features = []
        temporal_targets = (
            list(target_temporal_tokens)
            if target_temporal_tokens is not None
            else [None] * len(video_paths)
        )
        for video_path, temporal_target in zip(video_paths, temporal_targets, strict=False):
            clip = self._load_video_clip(
                video_path,
                target_temporal_tokens=temporal_target,
            ).to(device=device, dtype=dtype)
            encoded = self.teacher(clip)
            features.append(encoded.squeeze(0).float())
        return features


def attach_distillation_projector(
    model: nn.Module,
    student_dim: int,
    teacher_dim: int,
) -> nn.Linear:
    projector = getattr(model, "visual_distill_projector", None)
    if isinstance(projector, nn.Linear):
        if projector.in_features == student_dim and projector.out_features == teacher_dim:
            return projector

    projector = nn.Linear(student_dim, teacher_dim, bias=False)
    setattr(model, "visual_distill_projector", projector)
    return projector


def get_distillation_projector(model: nn.Module) -> nn.Linear:
    queue = [model]
    visited = set()
    while queue:
        current = queue.pop(0)
        if current is None or id(current) in visited:
            continue
        visited.add(id(current))
        projector = getattr(current, "visual_distill_projector", None)
        if isinstance(projector, nn.Linear):
            return projector
        queue.extend(
            getattr(current, attr, None)
            for attr in ("module", "model", "base_model")
        )
    raise AttributeError("Distillation projector has not been attached to the model.")


def compute_feature_loss(
    student_tokens: torch.Tensor,
    teacher_tokens: torch.Tensor,
    loss_type: str,
    normalize_features: bool,
) -> torch.Tensor:
    if normalize_features:
        student_tokens = F.normalize(student_tokens.float(), dim=-1)
        teacher_tokens = F.normalize(teacher_tokens.float(), dim=-1)
    else:
        student_tokens = student_tokens.float()
        teacher_tokens = teacher_tokens.float()

    if loss_type == "mse":
        return F.mse_loss(student_tokens, teacher_tokens)
    if loss_type == "cosine":
        return 1.0 - F.cosine_similarity(student_tokens, teacher_tokens, dim=-1).mean()
    raise ValueError(f"Unsupported distill_loss_type: {loss_type}")


def average_losses(losses: Iterable[torch.Tensor]) -> Optional[torch.Tensor]:
    losses = list(losses)
    if not losses:
        return None
    return torch.stack(losses).mean()
