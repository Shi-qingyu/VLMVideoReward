import logging
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import Trainer

from src.train.distill_vjepa import (
    VJepa2TeacherEncoder,
    align_teacher_tokens_to_student_shape,
    attach_distillation_projector,
    average_losses,
    compute_feature_loss,
    get_distillation_projector,
    select_student_features,
    save_qwen_visual_pca_batch,
    split_visual_tokens_with_shapes,
    _find_visual_module,
    _flatten_path_batches,
)

logger = logging.getLogger(__name__)


class Qwen3VLDistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.distill_enabled = bool(getattr(self.args, "distill_enable", False))
        self._last_distill_loss: Optional[torch.Tensor] = None
        self._last_task_loss: Optional[torch.Tensor] = None
        self._last_logged_step = -1
        self._last_distill_visualize_step: Optional[int] = None

        if not self.distill_enabled:
            self.teacher_encoder = None
            self.visual_merge_size = 2
            return

        teacher_arch = getattr(self.args, "distill_teacher_arch")
        teacher_ckpt = getattr(self.args, "distill_teacher_ckpt")
        teacher_image_size = int(getattr(self.args, "distill_teacher_image_size", 384))
        teacher_num_video_frames = int(
            getattr(self.args, "distill_teacher_num_video_frames", 16)
        )
        self.teacher_encoder = VJepa2TeacherEncoder(
            teacher_arch=teacher_arch,
            checkpoint_path=teacher_ckpt,
            image_size=teacher_image_size,
            num_video_frames=teacher_num_video_frames,
        )

        model_config = getattr(self.model, "config", None)
        student_dim = getattr(model_config, "hidden_size", None)
        if student_dim is None:
            input_embeddings = self.model.get_input_embeddings()
            student_dim = getattr(input_embeddings, "embedding_dim", None)
        if student_dim is None:
            raise ValueError("Could not infer the Qwen visual token width for distillation.")
        student_dim = int(student_dim)
        attach_distillation_projector(
            self.model,
            student_dim=student_dim,
            teacher_dim=self.teacher_encoder.teacher_dim,
        )

        image_processor = getattr(self.processing_class, "image_processor", None)
        self.visual_merge_size = int(getattr(image_processor, "merge_size", 2))

    def _current_distill_weight(self) -> float:
        base_weight = float(getattr(self.args, "distill_weight", 0.0))
        if base_weight <= 0.0:
            return 0.0

        start_steps = max(int(getattr(self.args, "distill_start_steps", 0)), 0)
        warmup_steps = max(int(getattr(self.args, "distill_warmup_steps", 0)), 0)
        step = max(int(getattr(self.state, "global_step", 0)), 0)

        if step < start_steps:
            return 0.0
        if warmup_steps == 0:
            return base_weight

        progress = min(max((step - start_steps + 1) / warmup_steps, 0.0), 1.0)
        return base_weight * progress

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs: bool = False,
        num_items_in_batch: Optional[torch.Tensor] = None,
    ):
        distill_image_paths = inputs.pop("distill_image_paths", None)
        distill_video_paths = inputs.pop("distill_video_paths", None)
        distill_video_metadatas = inputs.pop("distill_video_metadatas", None)
        distill_weight = self._current_distill_weight()
        can_run_distillation = self._should_run_distillation(
            inputs=inputs,
            distill_image_paths=distill_image_paths,
            distill_video_paths=distill_video_paths,
        )
        should_distill = distill_weight > 0.0 and can_run_distillation
        should_visualize = self._should_visualize_distill_batch(
            inputs=inputs,
            distill_video_paths=distill_video_paths,
        )

        captured_visual_outputs = []
        hook_handle = None
        should_capture = should_distill or should_visualize

        if should_capture:
            visual_module = _find_visual_module(model)

            def _capture_hook(_module, _args, output):
                captured_visual_outputs.append(output)

            hook_handle = visual_module.register_forward_hook(_capture_hook)

        try:
            outputs = model(**inputs)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        task_loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        total_loss = task_loss
        self._last_task_loss = task_loss.detach()
        self._last_distill_loss = None
        image_result = None
        video_result = None

        if should_capture:
            image_result, video_result = self._split_captured_visual_outputs(
                inputs=inputs,
                captured_visual_outputs=captured_visual_outputs,
            )

        if should_visualize:
            self._maybe_visualize_distill_batch(
                video_result=video_result,
                inputs=inputs,
                distill_video_paths=distill_video_paths,
                distill_video_metadatas=distill_video_metadatas,
            )

        if should_distill:
            distill_loss = self._compute_distill_loss(
                model=model,
                inputs=inputs,
                image_result=image_result,
                video_result=video_result,
                distill_image_paths=distill_image_paths,
                distill_video_paths=distill_video_paths,
                distill_video_metadatas=distill_video_metadatas,
            )
            if distill_loss is not None:
                self._last_distill_loss = distill_loss.detach()
                total_loss = total_loss + distill_weight * distill_loss

        if (
            self.model.training
            and self.state.global_step != self._last_logged_step
            and self.state.global_step % max(int(getattr(self.args, "logging_steps", 1)), 1) == 0
        ):
            logs = {"task_loss": float(self._last_task_loss.cpu())}
            if self.distill_enabled:
                logs["distill_weight"] = distill_weight
            if self._last_distill_loss is not None:
                logs["distill_loss"] = float(self._last_distill_loss.cpu())
            self.log(logs)
            self._last_logged_step = self.state.global_step

        return (total_loss, outputs) if return_outputs else total_loss

    def _should_run_distillation(
        self,
        inputs: dict[str, Any],
        distill_image_paths,
        distill_video_paths,
    ) -> bool:
        if not self.distill_enabled or self.teacher_encoder is None:
            return False

        has_images = bool(getattr(self.args, "distill_use_images", True)) and bool(
            inputs.get("pixel_values") is not None and _flatten_path_batches(distill_image_paths)
        )
        has_videos = bool(getattr(self.args, "distill_use_videos", True)) and bool(
            inputs.get("pixel_values_videos") is not None and _flatten_path_batches(distill_video_paths)
        )
        return has_images or has_videos

    def _should_visualize_distill_batch(
        self,
        inputs: dict[str, Any],
        distill_video_paths,
    ) -> bool:
        if not bool(getattr(self.args, "distill_visualize", False)):
            return False
        if not self.distill_enabled:
            return False
        if not self.is_world_process_zero():
            return False
        step = int(getattr(self.state, "global_step", 0))
        if self._last_distill_visualize_step == step:
            return False

        visualize_steps = int(getattr(self.args, "distill_visualize_steps", 0))
        if visualize_steps <= 0:
            if self._last_distill_visualize_step is not None:
                return False
        elif step % visualize_steps != 0:
            return False

        return bool(
            inputs.get("pixel_values_videos") is not None
            and inputs.get("video_grid_thw") is not None
            and _flatten_path_batches(distill_video_paths)
        )

    def _split_captured_visual_outputs(
        self,
        inputs: dict[str, Any],
        captured_visual_outputs: list[Any],
    ) -> tuple[Optional[Any], Optional[Any]]:
        image_result = None
        video_result = None
        output_index = 0
        if inputs.get("pixel_values") is not None:
            if output_index >= len(captured_visual_outputs):
                raise RuntimeError("Missing captured Qwen image visual outputs for distillation.")
            image_result = captured_visual_outputs[output_index]
            output_index += 1
        if inputs.get("pixel_values_videos") is not None:
            if output_index >= len(captured_visual_outputs):
                raise RuntimeError("Missing captured Qwen video visual outputs for distillation.")
            video_result = captured_visual_outputs[output_index]
        return image_result, video_result

    def _maybe_visualize_distill_batch(
        self,
        video_result: Any,
        inputs: dict[str, Any],
        distill_video_paths,
        distill_video_metadatas,
    ) -> None:
        if video_result is None:
            return

        output_dir = getattr(self.args, "distill_visualize_dir", None)
        if not output_dir:
            output_dir = str(Path(self.args.output_dir) / "distill_visualizations")

        step = int(getattr(self.state, "global_step", 0))
        self._last_distill_visualize_step = step
        try:
            saved_paths = save_qwen_visual_pca_batch(
                output_dir=output_dir,
                step=step,
                video_result=video_result,
                video_grid_thw=inputs.get("video_grid_thw"),
                distill_video_paths=distill_video_paths,
                distill_video_metadatas=distill_video_metadatas,
                feature_source=getattr(self.args, "distill_feature_source", "visual"),
                merge_size=self.visual_merge_size,
                max_items=int(getattr(self.args, "distill_visualize_max_items", 1)),
                max_frames=int(getattr(self.args, "distill_visualize_max_frames", 0)),
            )
        except Exception:
            logger.exception("Failed to save Qwen3VL visual PCA distillation preview.")
            return

        if saved_paths:
            logger.info(
                "Saved Qwen3VL visual PCA distillation preview for step %s to %s",
                step,
                output_dir,
            )

    def _compute_distill_loss(
        self,
        model,
        inputs: dict[str, Any],
        image_result: Any,
        video_result: Any,
        distill_image_paths,
        distill_video_paths,
        distill_video_metadatas,
    ) -> Optional[torch.Tensor]:
        if self.teacher_encoder is None:
            return None

        feature_source = getattr(self.args, "distill_feature_source", "visual")
        loss_type = getattr(self.args, "distill_loss_type", "mse")
        normalize_features = bool(
            getattr(self.args, "distill_normalize_features", True)
        )
        device = inputs["input_ids"].device
        model_dtype = next(model.parameters()).dtype
        teacher_dtype = model_dtype if device.type == "cuda" else torch.float32
        projector = get_distillation_projector(model).to(device=device)

        losses = []

        if (
            image_result is not None
            and getattr(self.args, "distill_use_images", True)
        ):
            image_paths = _flatten_path_batches(distill_image_paths)
            student_image_tokens = select_student_features(image_result, feature_source)
            student_image_groups = split_visual_tokens_with_shapes(
                student_image_tokens,
                inputs.get("image_grid_thw"),
                self.visual_merge_size,
            )
            teacher_image_groups = self.teacher_encoder.encode_images(
                image_paths=image_paths,
                device=device,
                dtype=teacher_dtype,
            )
            losses.extend(
                self._align_and_score(
                    projector=projector,
                    student_groups=student_image_groups,
                    teacher_groups=teacher_image_groups,
                    loss_type=loss_type,
                    normalize_features=normalize_features,
                    modality="image",
                )
            )

        if (
            video_result is not None
            and getattr(self.args, "distill_use_videos", True)
        ):
            video_paths = _flatten_path_batches(distill_video_paths)
            video_metadatas = _flatten_path_batches(distill_video_metadatas)
            student_video_tokens = select_student_features(video_result, feature_source)
            student_video_groups = split_visual_tokens_with_shapes(
                student_video_tokens,
                inputs.get("video_grid_thw"),
                self.visual_merge_size,
            )
            target_temporal_tokens = [shape[0] for _, shape in student_video_groups]
            teacher_video_groups = self.teacher_encoder.encode_videos(
                video_paths=video_paths,
                device=device,
                dtype=teacher_dtype,
                video_metadatas=video_metadatas if video_metadatas else None,
                target_temporal_tokens=target_temporal_tokens,
            )
            losses.extend(
                self._align_and_score(
                    projector=projector,
                    student_groups=student_video_groups,
                    teacher_groups=teacher_video_groups,
                    loss_type=loss_type,
                    normalize_features=normalize_features,
                    modality="video",
                )
            )

        return average_losses(losses)

    def _align_and_score(
        self,
        projector,
        student_groups,
        teacher_groups,
        loss_type: str,
        normalize_features: bool,
        modality: str,
    ) -> list[torch.Tensor]:
        if len(student_groups) != len(teacher_groups):
            raise ValueError(
                f"Student/teacher {modality} item count mismatch: "
                f"{len(student_groups)} vs {len(teacher_groups)}"
            )

        losses = []
        for (student_tokens, student_shape), teacher_tokens in zip(
            student_groups,
            teacher_groups,
            strict=False,
        ):
            aligned_teacher = align_teacher_tokens_to_student_shape(
                teacher_tokens,
                student_shape=student_shape,
                modality=modality,
            )
            projected_student = projector(student_tokens)
            losses.append(
                compute_feature_loss(
                    student_tokens=projected_student,
                    teacher_tokens=aligned_teacher.to(projected_student.device),
                    loss_type=loss_type,
                    normalize_features=normalize_features,
                )
            )
        return losses
