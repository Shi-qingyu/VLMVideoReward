import json
import re

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
    if x is None:
        return "fail"

    x = str(x).strip().lower().rstrip("。").rstrip(".")
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


def extract_tag_content(text: str, tag: str):
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text or "", re.S | re.I)
    return [match.strip() for match in matches] if matches else [""]


def parse_box(box_str: str):
    return json.loads(box_str)


def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter_area = inter_w * inter_h

    box1_area = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    box2_area = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def mean_matched_iou(gt_boxes, pred_boxes):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0

    iou_matrix = []
    for gt_box in gt_boxes:
        row = []
        for pred_box in pred_boxes:
            row.append(compute_iou(gt_box, pred_box))
        iou_matrix.append(row)

    cost_matrix = [[-iou for iou in row] for row in iou_matrix]
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    matched_ious = [iou_matrix[r][c] for r, c in zip(row_ind, col_ind)]
    return float(sum(matched_ious) / max(len(gt_boxes), len(pred_boxes)))


def parse_answer_block(answer_text: str):
    answer_dict = {}
    for key in EXPECTED_KEYS:
        pattern = rf"(?:^|\n)\s*{re.escape(key)}\s*:\s*([^\n]+)"
        match = re.search(pattern, answer_text or "", re.I)
        if match:
            answer_dict[key] = match.group(1).strip().rstrip(".").strip()
        else:
            answer_dict[key] = "Fail"
    return answer_dict


def parse_output(text: str):
    text = "" if text is None else str(text)
    think_content = extract_tag_content(text, "think")[0]
    answer_text = extract_tag_content(text, "answer")[0]
    answer_dict = parse_answer_block(answer_text)
    return think_content, answer_dict


def acc_reward(solution_str, ground_truth):
    _, pred_dict = parse_output(solution_str)
    _, gt_dict = parse_output(ground_truth)

    cnt = 0
    for key in EXPECTED_KEYS:
        if normalize_label(pred_dict[key]) == normalize_label(gt_dict[key]):
            cnt += 1
    return cnt / len(EXPECTED_KEYS)


def format_reward(solution_str):
    pattern = r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$"
    return 1.0 if re.match(pattern, str(solution_str), re.S | re.I) else 0.0


def iou_reward(solution_str, ground_truth):
    gt_boxes = extract_tag_content(ground_truth, "region")
    if gt_boxes[0] == "":
        return 1.0

    gt_boxes = [parse_box(box_str) for box_str in gt_boxes]
    pred_boxes = extract_tag_content(solution_str, "region")
    if pred_boxes[0] == "":
        return 0.0

    try:
        pred_boxes = [parse_box(box_str) for box_str in pred_boxes]
        return mean_matched_iou(gt_boxes, pred_boxes)
    except Exception:
        return 0.0


def compute_score(prompt, solution_str, ground_truth, extra_info=None):
    extra_info = extra_info or {}
    acc_weight = extra_info.get("acc_reward_weight", 1.0)
    format_weight = extra_info.get("format_reward_weight", 0.0)
    iou_weight = extra_info.get("iou_reward_weight", 0.0)

    acc_score = acc_reward(solution_str, ground_truth) * acc_weight
    format_score = format_reward(solution_str) * format_weight
    iou_score = iou_reward(solution_str, ground_truth) * iou_weight
    total_score = acc_score + format_score + iou_score
    return total_score, acc_score, format_score
