import re
import json
from typing import List, Tuple, Dict
from scipy.optimize import linear_sum_assignment


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
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text, re.S | re.I)
    return [match.strip() for match in matches] if matches else [""]


def parse_box(box_str: str) -> List[int]:
    """
    Parse a box string like "[x1,y1,x2,y2]" into a list of integers [x1, y1, x2, y2].
    """
    return json.loads(box_str)


def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    Box format: [x1, y1, x2, y2]
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Compute intersection area
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    # Compute area of each box
    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    # Compute union area
    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0

    return inter_area / union_area


def mean_matched_iou(gt_boxes, pred_boxes):
    """
    Compute mean IoU after optimal bipartite matching.

    The matching is solved with the Hungarian algorithm to maximize
    total IoU between ground-truth boxes and predicted boxes.

    To penalize unmatched boxes when the numbers do not match, the final
    reward is normalized by max(len(gt_boxes), len(pred_boxes)).
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    # Build IoU matrix of shape [num_gt, num_pred]
    iou_matrix = []
    for gt_box in gt_boxes:
        row = []
        for pred_box in pred_boxes:
            row.append(compute_iou(gt_box, pred_box))
        iou_matrix.append(row)

    # Hungarian algorithm minimizes cost, so use negative IoU as cost
    cost_matrix = [[-iou for iou in row] for row in iou_matrix]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matched_ious = [iou_matrix[r][c] for r, c in zip(row_ind, col_ind)]

    # Normalize by the larger number of boxes so unmatched boxes are penalized
    reward = sum(matched_ious) / max(len(gt_boxes), len(pred_boxes))
    return float(reward)


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

    think_content = extract_tag_content(text, "think")[0]
    answer_text = extract_tag_content(text, "answer")[0]
    answer_dict = parse_answer_block(answer_text)

    return think_content, answer_dict