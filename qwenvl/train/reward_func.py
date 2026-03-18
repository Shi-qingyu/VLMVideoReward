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
    x = re.split(r"[\s,;:，；：]+", x)[0] if x else "fail"

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
            value = value.rstrip("。").rstrip(".").strip()
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


def acc_reward(model_output: List[str], ground_truth: List[str], **kwargs) -> List[float]:
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

        ret.append(cnt / len(EXPECTED_KEYS))

    return ret


def format_reward(model_output: List[str], **kwargs) -> List[float]:
    """
    Reward for basic structural format:
      <think>...</think>
      <answer>...</answer>
    """
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    matches = [re.match(pattern, str(content), re.S | re.I) for content in model_output]
    return [0.2 if match else 0.0 for match in matches]


def field_reward(model_output: List[str], **kwargs) -> List[float]:
    """
    Reward for whether all expected answer fields appear.
    Returns a dense score in [0, 0.2].
    """
    rewards = []

    for output in model_output:
        _, answer_dict = parse_output(output)
        hit = 0
        for key in EXPECTED_KEYS:
            if answer_dict[key] != "Fail":
                hit += 1
        rewards.append(0.2 * hit / len(EXPECTED_KEYS))

    return rewards


def think_reward(model_output: List[str], **kwargs) -> List[float]:
    """
    Small reward for having non-empty <think>...</think>.
    """
    rewards = []
    for output in model_output:
        think_content, _ = parse_output(output)
        rewards.append(0.05 if len(think_content.strip()) > 0 else 0.0)
    return rewards