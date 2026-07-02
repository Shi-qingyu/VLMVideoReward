from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import Qwen3VLForConditionalGeneration
from transformers.cache_utils import Cache
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.models.qwen3_vl.modeling_qwen3_vl import apply_rotary_pos_emb
from transformers.processing_utils import Unpack
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train.argument import ModelArguments, SFTArguments
from src.train.sft_common import (
    SFTModelSpec,
    TrainableModulePaths,
    add_tokens_and_resize,
    train_sft,
)

logger = logging.get_logger(__name__)

QWEN_EXTRA_TOKENS = (
    "<think>",
    "</think>",
    "<answer>",
    "</answer>",
    "<box>",
    "</box>",
    "<t>",
    "</t>",
)


def _flash_attention_forward(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    dropout: float = 0.0,
    scaling: Optional[float] = None,
    sliding_window: Optional[int] = None,
    softcap: Optional[float] = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    del dropout, scaling, sliding_window, softcap
    if kwargs.get("output_attentions", False) or kwargs.get("head_mask") is not None:
        logger.warning_once(
            "`flash_attention_2` does not support `output_attentions=True` or `head_mask`. "
            "Use eager attention for those features."
        )
    if any(dim == 0 for dim in query.shape):
        raise ValueError("FlashAttention does not support zero-sized query tensors.")

    from flash_attn.flash_attn_interface import flash_attn_varlen_func

    query = query.transpose(1, 2).squeeze(0)
    key = key.transpose(1, 2).squeeze(0)
    value = value.transpose(1, 2).squeeze(0)
    cu_seqlens = attention_mask

    with torch.no_grad():
        max_seqlen = max(
            cu_seqlens[idx + 1] - cu_seqlens[idx]
            for idx in range(cu_seqlens.size(0) - 1)
        ).item()

    attn_output = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_k=cu_seqlens,
        max_seqlen_q=max_seqlen,
        max_seqlen_k=max_seqlen,
        causal=True,
    )
    return attn_output.unsqueeze(0), None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def _qwen3vl_packed_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs: Unpack[FlashAttentionKwargs],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            cache_kwargs,
        )

    attn_output, attn_weights = _flash_attention_forward(
        self,
        query_states,
        key_states,
        value_states,
        attention_mask,
        dropout=0.0 if not self.training else self.attention_dropout,
        scaling=self.scaling,
        **kwargs,
    )

    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, attn_weights


def _return_attention_mask(
    config,
    inputs_embeds=None,
    attention_mask=None,
    cache_position=None,
    past_key_values=None,
    position_ids=None,
    **kwargs,
):
    del config, inputs_embeds, cache_position, past_key_values, position_ids, kwargs
    return attention_mask


def replace_qwen3_vl_attention_class() -> None:
    from transformers.models.qwen3_vl import modeling_qwen3_vl
    from transformers.models.qwen3_vl_moe import modeling_qwen3_vl_moe

    modeling_qwen3_vl.Qwen3VLTextAttention.forward = _qwen3vl_packed_forward
    modeling_qwen3_vl.create_causal_mask = _return_attention_mask
    modeling_qwen3_vl_moe.Qwen3VLMoeTextAttention.forward = _qwen3vl_packed_forward
    modeling_qwen3_vl_moe.create_causal_mask = _return_attention_mask


def load_model(
    model_args: ModelArguments,
    training_args: SFTArguments,
    attn_implementation: str,
) -> torch.nn.Module:
    torch_dtype = torch.bfloat16 if training_args.bf16 else None
    return Qwen3VLForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        attn_implementation=attn_implementation,
        torch_dtype=torch_dtype,
    )


def load_processor(model_args: ModelArguments, training_args: SFTArguments):
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )


def prepare_processor(processor, model, data_args, training_args) -> None:
    del data_args, training_args
    add_tokens_and_resize(processor.tokenizer, model, QWEN_EXTRA_TOKENS)


def prepare_model_for_training(model, processor, data_args, training_args) -> None:
    del model, processor, training_args
    if data_args.data_flatten or data_args.data_packing:
        replace_qwen3_vl_attention_class()


QWEN_SPEC = SFTModelSpec(
    model_type="qwen3vl",
    default_attn_implementation="flash_attention_2",
    load_model=load_model,
    load_processor=load_processor,
    prepare_processor=prepare_processor,
    prepare_model_for_training=prepare_model_for_training,
    supports_flatten=True,
    trainable_paths=TrainableModulePaths(
        vision=("model.visual",),
        projector=("model.visual.merger",),
        language=("model.language_model",),
    ),
)


def train(attn_implementation: Optional[str] = None) -> None:
    train_sft(QWEN_SPEC, attn_implementation=attn_implementation)


if __name__ == "__main__":
    train(attn_implementation=os.environ.get("ATTN_IMPLEMENTATION", "flash_attention_2"))
