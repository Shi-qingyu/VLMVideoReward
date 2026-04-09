# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
import io
import os
import json
import copy
import math
import re
import torch
import random
import datasets
import numpy as np
import pandas as pd
from omegaconf import ListConfig
from collections import defaultdict
from torch.utils.data import Dataset
from typing import List, Union, Optional
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
from transformers import PreTrainedTokenizer, ProcessorMixin
from verl.utils.dataset.task_prompt import get_system_prompt, process_problem_with_data_source
from my_qwen_vl_utils.vision_process import fetch_video_raw, resample_video_from_raw, fetch_video_raw_frame
import pickle

datasets.disable_caching()


def _resolve_media_path(path_value: str, data_file_dir: Optional[str] = None, media_root: Optional[str] = None):
    if not path_value:
        return path_value
    if os.path.isabs(path_value):
        return path_value
    if media_root:
        return os.path.join(media_root, path_value)
    if data_file_dir:
        return os.path.join(data_file_dir, path_value)
    return path_value


def _build_chat_from_qwenvl_conversations(row, media_root: Optional[str] = None):
    conversations = row["conversations"]
    if not conversations:
        raise ValueError("Empty conversations field.")

    images = list(row.get("images") or [])
    videos = list(row.get("videos") or [])
    if isinstance(images, str):
        images = [images]
    if isinstance(videos, str):
        videos = [videos]

    data_file_dir = row.get("__data_file_dir")
    image_pool = [
        {"type": "image", "image": _resolve_media_path(img, data_file_dir=data_file_dir, media_root=media_root)}
        for img in images
    ]
    video_pool = [
        {"type": "video", "video": _resolve_media_path(video, data_file_dir=data_file_dir, media_root=media_root)}
        for video in videos
    ]

    messages = []
    for turn in conversations:
        role = "user" if turn["from"] == "human" else "assistant"
        text = turn["value"]

        if role == "user":
            content = []
            for seg in re.split(r"(<image>|<video>)", text):
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError("Number of <image> placeholders exceeds provided images.")
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError("Number of <video> placeholders exceeds provided videos.")
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})
            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool or video_pool:
        raise ValueError("Unused images or videos remain after parsing placeholders.")

    return messages


def _extract_qwenvl_prompt_and_answer(row, media_root: Optional[str] = None):
    messages = _build_chat_from_qwenvl_conversations(row, media_root=media_root)
    user_messages = [message for message in messages if message["role"] == "user"]
    assistant_messages = [message for message in messages if message["role"] == "assistant"]
    if not user_messages or not assistant_messages:
        raise ValueError("Expected at least one user turn and one assistant turn in conversations.")

    answer_text = assistant_messages[-1]["content"][0]["text"]
    return user_messages, answer_text

def make_conversation_multimodal_video(df, system_prompt=None):
    def make_conv(row):
        if "conversations" in row and isinstance(row["conversations"], list):
            user_messages, answer_text = _extract_qwenvl_prompt_and_answer(row)
            row["prompt"] = user_messages
            ground_truth = row.get("ground_truth", None)
            row["ground_truth"] = answer_text if ground_truth is None or pd.isna(ground_truth) else ground_truth
            if "data_source" not in row or pd.isna(row["data_source"]):
                row["data_source"] = row.get("__default_data_source", "qwenvl_video_judge")
            return row

        # Apply data source specific prompt processing
        system_prompt = get_system_prompt(row)
        processed_problem = process_problem_with_data_source(row)
        prompt = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": processed_problem,
            },
        ]
        # Handle video data (video field instead of images)
        if "video" in row:
            video = row["video"]
            base_image_dir = os.getenv("BASE_IMAGE_DIR", "./data")
            row["video"] = os.path.join(base_image_dir, video)
        row["prompt"] = prompt
        return row
    df = df.apply(make_conv, axis=1)
    return df

def collate_fn(data_list: list[dict]) -> dict:
    tensors = defaultdict(list)
    non_tensors = defaultdict(list)

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
            else:
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        if key == 'raw_prompt_ids': # FIXME: hack to cope with dataset whose input_ids is all the same
            array_list = np.empty(len(val), dtype=object)
            for i, lst in enumerate(val):
                array_list[i] = lst
            non_tensors[key] = array_list
        else:
            non_tensors[key] = np.array(val, dtype=object)

    return {**tensors, **non_tensors}


def process_video_adaptive_token_num(video_path: str, max_tokens: int = 768 * 28 * 28, min_tokens: int = 128 * 28 * 28, fps: float = 2.0, source_frames_fps: float = 4.0):
    """
    Process video file using my_qwen_vl_utils.vision_process.fetch_video_raw and resample_video_from_raw
    
    Args:
        video_path: Path to the video file
        max_tokens: Maximum tokens per frame (default for video)
        min_tokens: Minimum tokens per frame (default for video)
    
    Returns:
        tuple: (processed_video_tensor, raw_video_tensor, sample_fps)
            - processed_video_tensor: Resampled video tensor with shape (T, C, H, W)
            - raw_video_tensor: Raw video tensor with shape (T, C, H, W)
            - sample_fps: Sample fps from fetch_video_raw
    """
    assert os.path.exists(video_path), f"Video path {video_path} not exists."
    
    try:
        # Create video element dict for fetch_video_raw
        video_ele = {
            "video": video_path,
            "fps": fps,
            "source_frames_fps": source_frames_fps,
        }
        # Fetch raw video and fps
        if video_path.endswith(".mp4"):
            raw_video, raw_sample_fps = fetch_video_raw(video_ele)
        else:
            raw_video, raw_sample_fps = fetch_video_raw_frame(video_ele)
        
        frames_num=raw_video.shape[0]
        max_tokens_per_frame = max_tokens // (frames_num//2)
        min_tokens_per_frame = min_tokens // (frames_num//2)
        # Create video element dict for resample_video_from_raw
        resample_ele = {
            "max_pixels": max_tokens_per_frame * 28 * 28,
            "min_pixels": min_tokens_per_frame * 28 * 28,
        }
        # Resample and resize video
        video, sample_fps = resample_video_from_raw(
            raw_video,
            raw_sample_fps,
            resample_ele,
            return_video_sample_fps=True
        )
        # print(f"video: {video.shape}, fps: {sample_fps}, raw_video: {raw_video.shape}, raw_sample_fps: {raw_sample_fps}, max_tokens_per_frame: {max_tokens_per_frame}, min_tokens_per_frame: {min_tokens_per_frame}")
        return video, sample_fps, raw_video, raw_sample_fps
    except Exception as exn:
        print(f"Failed to process video {video_path}. Exception:", exn)
        raise exn


class MultiModalDatasetVideo(Dataset):
    """
    Dataset for video data loading. Similar to MultiModalDataset but handles video instead of images.
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 data_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 processor: Optional[ProcessorMixin] = None,
                 prompt_key='prompt',
                 answer_key='answer',
                 video_key='video',
                 max_prompt_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 system_prompt: str = None,
                 max_pixels: int = 768 * 28 * 28,  # Default for video
                 min_pixels: int = 128 * 28 * 28,  # Default for video
                 mask_blank: bool = False,
                 use_3drope: bool = True,
                 general_qa_reward_fn: str = 'v1',
                 use_raw_image: bool = False,
                 media_root: Optional[str] = None,
                 default_data_source: str = "qwenvl_video_judge",
                ):
        import warnings
        warnings.filterwarnings("ignore")
        import logging
        logging.disable(logging.CRITICAL)

        if not isinstance(data_files, (List, ListConfig)):
            data_files = [data_files]

        self.data_files = copy.deepcopy(data_files)
        self.original_data_files = copy.deepcopy(data_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        self.processor = processor

        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.video_key = video_key
        self.max_prompt_length = max_prompt_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation

        self.system_prompt = system_prompt
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        
        self.mask_blank = mask_blank
        self.use_3drope = use_3drope
        self.general_qa_reward_fn = general_qa_reward_fn
        self.use_raw_image = use_raw_image
        self.media_root = media_root
        self.default_data_source = default_data_source

        print("self.use_3drope: ", self.use_3drope)
        print("self.general_qa_reward_fn: ", self.general_qa_reward_fn)
        print("self.use_raw_image:", self.use_raw_image)

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        # self._download()
        self._read_files_and_tokenize()

    def _download(self, use_origin_data_files=False):
        from verl.utils.fs import copy_to_local
        data_files = self.data_files if not use_origin_data_files else self.original_data_files
        for i, data_file in enumerate(data_files):
            self.data_files[i] = copy_to_local(src=data_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for data_file in self.data_files:
            with open(data_file, "r") as f:
                json_data = json.load(f)
            dataframe = pd.DataFrame(json_data)
            dataframe["__data_file_dir"] = os.path.dirname(os.path.abspath(data_file))
            dataframe["__default_data_source"] = self.default_data_source
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        print(f">>> Original Data Size: {len(self.dataframe)}")

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = make_conversation_multimodal_video(self.dataframe, self.system_prompt)
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length, axis=1)]

        print(f'>>> Filtered Dataset Size: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_arrow_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_arrow=True)  # download and resume from original arrow files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')
    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        chat = row_dict.pop(self.prompt_key)
        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        
        video_value = row_dict.get(self.video_key)
        if isinstance(video_value, list):
            if len(video_value) != 1:
                raise ValueError(f"Only single-video samples are supported, got {len(video_value)} videos.")
            video_value = video_value[0]
        if isinstance(video_value, str) and not os.path.isabs(video_value):
            video_value = _resolve_media_path(
                video_value,
                data_file_dir=row_dict.get("__data_file_dir"),
                media_root=self.media_root,
            )
        if video_value is not None:
            row_dict[self.video_key] = video_value

        if self.video_key in row_dict:  # expand video token
            video_path = row_dict[self.video_key]
            try:
                # Process video using fetch_video_raw and resample_video_from_raw
                video_tensor, sample_fps, raw_video_tensor, raw_sample_fps = process_video_adaptive_token_num(video_path, self.max_pixels, self.min_pixels, fps=float(os.environ.get('SELF_SET_OVERVIEW_FPS', 2.0)), source_frames_fps=row_dict.get('frame_fps', 4.0))
                # Save processed video and raw video with fps
                row_dict['multi_modal_data'] = {'video': [video_tensor]}
                row_dict['video_fps_used'] = {'fps':[sample_fps]}
                if video_path.endswith(".mp4"):
                    row_dict['raw_multi_modal_metadata'] = {
                        'video': {'path': video_path, 'tensor': raw_video_tensor},
                        'fps': raw_sample_fps,
                        'length': raw_video_tensor.shape[0] / raw_sample_fps
                    }
                else:
                    row_dict['raw_multi_modal_metadata'] = {
                        'video': {'path': video_path, 'tensor': torch.zeros(0, dtype=torch.uint8)},
                        'fps': row_dict.get('frame_fps', 4.0),
                        'length': raw_video_tensor.shape[0] / raw_sample_fps
                    }
                time_instruction = f"This video is uniformly sampled at {(sample_fps):.2f} fps, contains {video_tensor.shape[0]} frames from 0 seconds to {(video_tensor.shape[0]/sample_fps-0.05):.1f} seconds."
            except Exception as e:
                print(str(e))
                return self.__getitem__(item+1) if item + 1 < len(self) else self.__getitem__(0)
            
            raw_prompt = prompt_with_chat_template.replace('<video>', '<|vision_start|><|video_pad|><|vision_end|>' + time_instruction)
            video_inputs = self.processor.video_processor(row_dict['multi_modal_data']['video'])
            video_grid_thw = video_inputs.get('video_grid_thw', None)
            merge_length = self.processor.video_processor.merge_size**2

            index = 0
            while '<video>' in prompt_with_chat_template:
                prompt_with_chat_template = prompt_with_chat_template.replace('<video>', '<|vision_start|>' + '<|video_pad|>' * (video_grid_thw[index].prod() // merge_length) + '<|vision_end|>' + time_instruction, 1)
                index += 1
        else:
            print(f"<<<WARNING>>> No video found in the prompt of {row_dict['doc_id']}! Please check the data json!")
            video_grid_thw = None 
            raw_prompt = prompt_with_chat_template

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        if self.mask_blank and self.video_key in row_dict and video_path.split("/")[-1].split(".")[0] == "blank_video":
            image_token_id = self.tokenizer.encode(self.processor.image_token)[0]
            attention_mask[input_ids == image_token_id] = 0

        if self.use_3drope and self.video_key in row_dict and video_grid_thw is not None:
            from verl.models.transformers.qwen2_vl import get_rope_index
            position_ids = [
                get_rope_index(
                    self.processor,
                    input_ids=input_ids[0],
                    image_grid_thw=None,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask[0],
                )
            ]  # (1, 3, seq_len)
            # Add raw video at the end of multimodal_dataset to avoid <|image_pad|> error
            row_dict.pop(self.video_key)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]
        row_dict['raw_prompt_ids'] = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        row_dict['data_source'] = row_dict.get('data_source', self.default_data_source)
        if 'reward_model' not in row_dict:
            # Handle both 'solution' and 'answer' fields
            if 'ground_truth' in row_dict:
                pass
            elif 'solution' in row_dict:
                row_dict['ground_truth'] = row_dict.pop('solution')
            elif 'answer' in row_dict:
                row_dict['ground_truth'] = row_dict.pop('answer')
        # encode prompts without chat template
        if self.return_raw_chat:
            if isinstance(chat[-1]['content'], list):
                row_dict['raw_prompt'] = "".join(
                    item.get("text", "<video>" if item.get("type") == "video" else "<image>")
                    for item in chat[-1]['content']
                )
            else:
                assert chat[-1]['role'] == 'user'
                row_dict['raw_prompt'] = chat[-1]['content']

        # add index for each prompt
        index = row_dict.get("doc_id", row_dict.get("id", 0))
        row_dict["index"] = index
        row_dict.pop("options", None)
        row_dict.pop("__data_file_dir", None)
        row_dict.pop("__default_data_source", None)
        row_dict.pop("conversations", None)

        try:
            pickle.dumps(row_dict)
        except Exception as e:
            print("Pickle error:", e)

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
