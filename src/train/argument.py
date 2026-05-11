import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")


@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    model_type: str = field(default="qwen3vl")
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    gemma4_max_soft_tokens: Optional[int] = field(default=None)
    minicpmv_max_slice_nums: Optional[int] = field(default=None)
    minicpmv_video_group_size: int = field(default=6)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2
    using_cot: bool = field(default=True)


@dataclass
class SFTArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

    # Visual distillation config
    distill_enable: bool = field(default=False)
    distill_teacher_arch: str = field(default="vjepa2_1_vit_large_384")
    distill_teacher_ckpt: str = field(default="vjepa2_1_vitl_dist_vitG_384.pt")
    distill_weight: float = field(default=1.0)
    distill_start_steps: int = field(default=0)
    distill_warmup_steps: int = field(default=0)
    distill_loss_type: str = field(default="mse")
    distill_feature_source: str = field(default="visual")
    distill_normalize_features: bool = field(default=True)
    distill_teacher_image_size: int = field(default=384)
    distill_teacher_num_video_frames: int = field(default=16)
    distill_use_images: bool = field(default=True)
    distill_use_videos: bool = field(default=True)
    distill_visualize: bool = field(default=False)
    distill_visualize_dir: Optional[str] = field(default=None)
    distill_visualize_steps: int = field(default=0)
    distill_visualize_max_items: int = field(default=1)
    distill_visualize_max_frames: int = field(default=0)


@dataclass
class GRPOArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    max_input_length: int = field(
        default=16384,
        metadata={
            "help": "Maximum input sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    # Sampling config
    max_new_tokens: int = field(default=256)
    num_generations: int = field(default=8)
    top_p: float = field(default=0.95)
    temperature: float = field(default=1.0)

    # KL penalty
    beta: float = field(default=0.04)

    # Reward function
    reward_func: str = field(default="acc_reward")
    reward_func_weight: str = field(default="1.0")

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)
