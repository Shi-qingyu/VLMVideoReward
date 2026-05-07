import re
import json
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment

from utils import *


def acc_reward(
    model_output: List[str], 
    ground_truth: List[str], 
    model_input: List[str],
    **kwargs
) -> List[float]:
    """
    Accuracy reward over the 7 fixed fields.
    Each sample gets score in [0, 1].
    """
    ret = []

    for output, gt in zip(model_output, ground_truth):
        _, pred_dict = parse_output(output)
        _, gt_dict = parse_output(gt)

        cnt = 0
        for key in EXPECTED_KEYS:
            pred_val = normalize_label(pred_dict[key])
            gt_val = normalize_label(gt_dict[key])
            if pred_val == gt_val:
                cnt += 1

        reward = cnt / len(EXPECTED_KEYS)
        ret.append(reward)

    return ret


def format_reward(
    model_output: List[str], 
    ground_truth: List[str], 
    model_input: List[str],
    **kwargs
) -> List[float]:
    """
    Reward for basic structural format:
      <think>...</think>
      <answer>...</answer>
    """
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    matches = [re.match(pattern, str(content), re.S | re.I) for content in model_output]
    return [1.0 if match else 0.0 for match in matches]


def iou_reward(
    model_output: List[str],
    ground_truth: List[str],
    model_input: List[str],
    **kwargs
) -> List[float]:
    """
    Region reward based on the mean IoU of optimally matched boxes.
    """
    rewards = []

    for out, gt in zip(model_output, ground_truth):
        gt_boxes = extract_tag_content(gt, "region")

        if gt_boxes[0] != "":
            gt_boxes = [parse_box(box_str) for box_str in gt_boxes]
            model_output_boxes = extract_tag_content(out, "region")

            if model_output_boxes[0] != "":
                try:
                    model_output_boxes = [parse_box(box_str) for box_str in model_output_boxes]

                    # Compute scalar reward from optimally matched box IoUs
                    reward = mean_matched_iou(gt_boxes, model_output_boxes)

                except Exception as e:
                    reward = 0.0
            else:
                reward = 0.0
        else:
            reward = 1.0

        rewards.append(reward)

    return rewards


def pseudo_reward(
    model_output: List[str], 
    ground_truth: List[str], 
    model_input: List[str],
    **kwargs
) -> List[float]:
    """
    pseudo_reward
    """
    return [1.0 for _ in model_output]