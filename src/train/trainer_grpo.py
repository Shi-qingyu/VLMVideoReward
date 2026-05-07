# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from typing import Any, Callable, Optional, Union
import random
import copy

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from qwenvl.train.argument import GRPOArguments
from qwenvl.train.trainer_sft import create_optimizer

from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb


# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def token_length_reward_from_tensor(
    token_lengths: torch.Tensor,
    min_len: int = 40,
    target_len: int = 80,
    max_len: int = 140,
    max_reward: float = 0.12,
):
    rewards = torch.zeros_like(token_lengths, dtype=torch.float32)

    inc_mask = (token_lengths >= min_len) & (token_lengths <= target_len)
    dec_mask = (token_lengths > target_len) & (token_lengths <= max_len)

    rewards[inc_mask] = max_reward * (token_lengths[inc_mask] - min_len) / max(1, target_len - min_len)
    rewards[dec_mask] = max_reward * (max_len - token_lengths[dec_mask]) / max(1, max_len - target_len)

    return torch.clamp(rewards, min=0.0)


class Qwen3VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        reward_func_weights: list[float],
        args: GRPOArguments = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        data_collator: Optional[Callable] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer],
            Optional[torch.optim.lr_scheduler.LambdaLR],
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Models
        model_init_kwargs = {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        model_init_kwargs["dtype"] = (torch.bfloat16 if args.bf16 else None)
        model_init_kwargs["cache_dir"] = args.cache_dir

        if isinstance(model, str):
            model_id = model
            if "qwen3" in model_id.lower():
                model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                raise ValueError(f"Unsupported model_id: {model_id}")
        else:
            model_id = model.config._name_or_path

        if args.gradient_checkpointing:
            assert peft_config is None, "gradient checkpointing is not compatiable with lora!"
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        if peft_config is not None:
            model = get_peft_model(model, peft_config)
        else:
            set_model(args, model)

        # Reference model
        if is_deepspeed_zero3_enabled():
            if "qwen3" in model_id.lower():
                self.ref_model = Qwen3VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
            else:
                raise ValueError(f"Unsupported model_id: {model_id}")
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        # Processing class
        if processing_class is None:
            processing_class = AutoProcessor.from_pretrained(model_id)
            pad_token_id = processing_class.tokenizer.pad_token_id
            processing_class.pad_token_id = pad_token_id
            processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
            processing_class.tokenizer.padding_side = "left"

            if hasattr(processing_class, "image_processor"):
                processing_class.image_processor.max_pixels = max_pixels
                processing_class.image_processor.min_pixels = min_pixels
        else:
            pad_token_id = processing_class.tokenizer.pad_token_id
            if hasattr(processing_class, "tokenizer"):
                processing_class.tokenizer.padding_side = "left"

        self.processing_class = processing_class

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]

        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func,
                    num_labels=1,
                    **model_init_kwargs,
                )
        self.reward_funcs = reward_funcs
        self.reward_func_weights = reward_func_weights
        assert len(self.reward_funcs) == len(self.reward_func_weights)

        # Reward processing classes
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        elif len(reward_processing_classes) != len(reward_funcs):
            raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class

        self.reward_processing_classes = reward_processing_classes

        # Training arguments
        self.max_input_length = args.max_input_length
        self.max_new_tokens = args.max_new_tokens
        self.num_generations = args.num_generations
        self.beta = args.beta

        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_new_tokens,
            do_sample=True,
            top_p=args.top_p,
            temperature=args.temperature,
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        # Metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model,
                    evaluation_mode=True,
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func,
                    evaluation_mode=True,
                )


    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["user", "gt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, **kwargs):
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        # import pdb
        # pdb.set_trace()
        logits = model(input_ids, **kwargs).logits.clone()
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs


    def _extract_text_from_message_list(self, messages):
        parts = []
        for msg in messages:
            content = msg.get("content", [])
            if isinstance(content, list):
                for c in content:
                    if isinstance(c, dict) and c.get("type") == "text":
                        parts.append(c.get("text", ""))
        return "".join(parts)


    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        device = self.accelerator.device

        # collator output:
        # {
        #   "user": [sample_user_messages, ...],
        #   "gt":   [sample_gt_messages, ...]
        # }
        user_messages = [copy.deepcopy(x) for x in inputs["user"]]
        gt_messages = [copy.deepcopy(x) for x in inputs["gt"]]

        batch_size = len(user_messages)

        # build model inputs directly from chat template
        # all data are video samples
        try:
            inputs = self.processing_class.apply_chat_template(
                user_messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=True,
            )
        except Exception as e:
            print(f"apply_chat_template failed: {e}")
            for idx, msg in enumerate(user_messages):
                print(f"bad sample {idx}: {msg}")
            raise

        # converts to tensor and move to device
        inputs = super()._prepare_inputs(inputs)

        # we pad on the left
        if self.max_input_length is not None:
            inputs["input_ids"] = inputs["input_ids"][:, -self.max_input_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -self.max_input_length:]

        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
    
        # generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            model_output_ids = unwrapped_model.generate(
                **inputs,
                generation_config=self.generation_config,
            )   # [B * G, L]

        prompt_length = input_ids.size(1)
        model_answer_ids = model_output_ids[:, prompt_length:]
        attention_mask_answer = torch.ones_like(model_answer_ids, device=device)
        attention_mask_for_logps = attention_mask.repeat_interleave(self.num_generations, dim=0)
        attention_mask_for_logps = torch.cat([attention_mask_for_logps, attention_mask_answer], dim=1)

        num_total = model_answer_ids.size(0)
        answer_length = model_answer_ids.size(1)

        # mask everything after first EOS
        eos_token_id = self.processing_class.tokenizer.eos_token_id
        is_eos = model_answer_ids == eos_token_id
        eos_idx = torch.full(
            (num_total,),
            answer_length,  # default to answer_length, meaning no EOS found
            dtype=torch.long,
            device=device,
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(answer_length, device=device).expand(num_total, -1)
        model_output_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()    # [num_total, answer_length]

        # prepare kwargs for forward logp computation
        inputs_for_logps = {
            k: v for k, v in inputs.items()
            if k not in ["input_ids"]
        }

        # expand video tensors from B -> B * G
        inputs_for_logps["attention_mask"] = attention_mask_for_logps
        
        if "pixel_values_videos" in inputs_for_logps:
            inputs_for_logps["pixel_values_videos"] = inputs_for_logps["pixel_values_videos"].repeat_interleave(
                self.num_generations, dim=0
            )

        if "video_grid_thw" in inputs_for_logps:
            inputs_for_logps["video_grid_thw"] = inputs_for_logps["video_grid_thw"].repeat_interleave(
                self.num_generations, dim=0
            )

        if "second_per_grid_ts" in inputs_for_logps:
            # remove unless you know exact repeat rule for your processor output
            del inputs_for_logps["second_per_grid_ts"]

        # compute model logps
        try:
            per_token_logps = self._get_per_token_logps(
                model,
                model_output_ids,
                **inputs_for_logps,
            )
            per_token_logps = per_token_logps[:, prompt_length-1:]
        except Exception as e:
            print(f"Error computing per_token_logps with video kwargs: {e}. Fallback to text-only forward.")
            per_token_logps = self._get_per_token_logps(model, model_output_ids)
            per_token_logps = per_token_logps[:, prompt_length-1:]

        # compute reference logps
        with torch.inference_mode():
            try:
                if self.ref_model is not None:
                    ref_per_token_logps = self._get_per_token_logps(
                        self.ref_model,
                        model_output_ids,
                        **inputs_for_logps,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps = self._get_per_token_logps(
                            model,
                            model_output_ids,
                            **inputs_for_logps,
                        )
                ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]
            except Exception as e:
                print(f"Error computing ref_per_token_logps with video kwargs: {e}. Fallback to text-only forward.")
                with self.accelerator.unwrap_model(model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(model, model_output_ids)
                ref_per_token_logps = ref_per_token_logps[:, prompt_length-1:]

        # KL divergence
        x_clamped = torch.clamp(ref_per_token_logps - per_token_logps, min=-10, max=10)
        per_token_kl = torch.exp(x_clamped) - x_clamped - 1

        # decode text
        model_answer_raw_text = self.processing_class.batch_decode(
            model_answer_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        # build reward inputs
        gt_answers_raw_text = [self._extract_text_from_message_list(x) for x in gt_messages]
        expanded_gt_answers_raw_text = [gt for gt in gt_answers_raw_text for _ in range(self.num_generations)]

        model_input_raw_text = [
            self.processing_class.apply_chat_template(
                msg,
                tokenize=False,
                add_generation_prompt=True,
            )
            for msg in user_messages
        ]
        expanded_model_input_raw_text = [p for p in model_input_raw_text for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(
            len(expanded_model_input_raw_text),
            len(self.reward_funcs),
            device=device,
        )

        for i, (reward_func, weight) in enumerate(zip(self.reward_funcs, self.reward_func_weights)):
            output_reward_func = reward_func(
                model_input=expanded_model_input_raw_text,
                model_output=model_answer_raw_text,
                ground_truth=expanded_gt_answers_raw_text,
            )
            rewards_per_func[:, i] = torch.tensor(
                output_reward_func,
                dtype=torch.float32,
                device=device,
            ) * weight

        rewards = rewards_per_func.sum(dim=1)

        # token_lengths = model_output_mask.sum(dim=1).float()
        # length_rewards = token_length_reward_from_tensor(
        #     token_lengths,
        #     min_len=80,
        #     target_len=120,
        #     max_len=160,
        #     max_reward=0.2,
        # )
    
        # gathered_length_rewards = self.accelerator.gather_for_metrics(length_rewards)
        # self._metrics["rewards/length_reward"].append(gathered_length_rewards.mean().item())

        # grouped rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)

        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # GRPO loss
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)

        loss = ((per_token_loss * model_output_mask).sum(dim=1) / model_output_mask.sum(dim=1)).mean()

        # logging
        model_output_length = self.accelerator.gather_for_metrics(model_output_mask.sum(1)).float().mean().item()
        self._metrics["model_output_length"].append(model_output_length)

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )

        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )

        mean_kl = ((per_token_kl * model_output_mask).sum(dim=1) / model_output_mask.sum(dim=1)).mean()
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item()
        )

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()
    
    
Qwen3VLGRPOTrainer.create_optimizer = create_optimizer