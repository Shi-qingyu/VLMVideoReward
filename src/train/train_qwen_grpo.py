# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import logging
import pathlib
import torch
import transformers
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)
from qwenvl.dataset.data_processor import make_rl_data_module
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    GRPOArguments,
)
from qwenvl.train.trainer_grpo import Qwen3VLGRPOTrainer
from qwenvl.train import reward_func as reward_module
from transformers import AutoProcessor, Trainer

local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def load_reward_funcs(reward_str):
    reward_names = [name.strip() for name in reward_str.split(",")]
    
    reward_funcs = []
    for name in reward_names:
        if not hasattr(reward_module, name):
            raise ValueError(f"Reward function {name} not found in the file qwenvl.train.reward_func.py :(")
        reward_funcs.append(getattr(reward_module, name))
    
    return reward_funcs


def train(attn_implementation="flash_attention_2"):
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, GRPOArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    reward_funcs = load_reward_funcs(training_args.reward_func)
    reward_func_weights = [float(w) for w in training_args.reward_func_weight.split(",")]

    local_rank = training_args.local_rank
    os.makedirs(training_args.output_dir, exist_ok=True)    

    if training_args.lora_enable:
        assert not training_args.gradient_checkpointing, "gradient checkpointing is not compatiable with lora!"
        from peft import LoraConfig, get_peft_model, TaskType
        print("LoRA enabled")

        peft_config = LoraConfig(
            r=training_args.lora_r or 64,
            lora_alpha=training_args.lora_alpha or 128,
            lora_dropout=training_args.lora_dropout or 0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
    else:
        peft_config = None

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
    )
    data_module = make_rl_data_module(processor, data_args=data_args)
    trainer = Qwen3VLGRPOTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        peft_config=peft_config,
        reward_funcs=reward_funcs,
        reward_func_weights=reward_func_weights,
        processing_class=processor,
        **data_module
    )

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        logging.info("checkpoint found, resume training")
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
    
    processor.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    train(attn_implementation="flash_attention_2")
