import re
from typing import List, Tuple, Dict


EXPECTED_KEYS = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
]


def normalize_label(x: str) -> str:
    """
    Normalize model / gt labels to a comparable canonical form.
    """
    if x is None:
        return "fail"

    x = str(x).strip().lower().rstrip("。").rstrip(".")

    # keep only the leading semantic label when model generates extra words
    # e.g. "yes, because ..." -> "yes"
    #      "good alignment"   -> "good"
    x = re.split(r"[\s,;:]+", x)[0] if x else "fail"

    yes_set = {
        "yes", "good", "true", "correct", "present", "exists", "aligned", "match", "matched", "plausible"
    }
    no_set = {
        "no", "bad", "false", "incorrect", "absent", "missing", "misaligned", "mismatch", "implausible"
    }

    if x in yes_set:
        return "yes"
    if x in no_set:
        return "no"
    return x if x else "fail"


def extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content inside <tag>...</tag>. Return empty string if not found.
    """
    if text is None:
        return ""
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    match = re.search(pattern, text, re.S | re.I)
    return match.group(1).strip() if match else ""


def parse_answer_block(answer_text: str) -> Dict[str, str]:
    """
    Parse the content inside <answer>...</answer> into a fixed dict.
    Missing keys are filled with 'Fail'.
    """
    answer_dict = {}

    for key in EXPECTED_KEYS:
        # Match one line like:
        # Video Quality: Yes
        # Cause-Effect : No
        #
        # Capture until end-of-line
        pattern = rf"(?:^|\n)\s*{re.escape(key)}\s*:\s*([^\n]+)"
        match = re.search(pattern, answer_text, re.I)
        if match:
            value = match.group(1).strip()
            value = value.rstrip(".").strip()
            answer_dict[key] = value
        else:
            answer_dict[key] = "Fail"

    return answer_dict


def parse_output(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Parse a full model output / gt string into:
      - think_content
      - answer_dict with fixed EXPECTED_KEYS
    """
    text = "" if text is None else str(text)

    think_content = extract_tag_content(text, "think")
    answer_text = extract_tag_content(text, "answer")
    answer_dict = parse_answer_block(answer_text)

    return think_content, answer_dict


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

        if cnt == 7:
            reward = 1.0
        elif cnt == 6:
            reward = 0.8
        elif cnt == 5:
            reward = 0.6
        elif cnt == 4:
            reward = 0.4
        else:
            reward = 0

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