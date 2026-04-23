import re
import json
from typing import Optional, Dict, Any, List

question = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\n"
    "Evaluate the video according to the following dimensions.\n"
    "Video Quality: whether the video is free from major visual defects, including blur, lack of detail, "
    "poor texture, lighting issues, color distortion, flickering, and overexposure.\n"
    "Subject Movement: whether the subject's motion is natural, smooth, and realistic.\n"
    "Physical Interaction: whether interactions among subjects and/or objects are physically plausible.\n"
    "Cause-Effect: whether causal relationships are correctly depicted.\n"
    "Subject Existence: whether the subject described in the prompt appears and is accurate.\n"
    "Object Existence: whether the object described in the prompt appears and is accurate.\n"
    "Subject-Object Interaction: whether the interaction described in the prompt is correctly represented.\n"
    "Prompt: {prompt} Provide your reasoning trace between think tags <think> and </think>, "
    "then output \"Yes\" or \"No\" for each dimension between <answer> and </answer>."
)


def split_sentences(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def should_drop_sentence(sent: str) -> bool:
    s = sent.lower().strip()

    drop_patterns = [
        r"\bai-generated\b",
        r"\bai generated\b",
        r"\bi believe the video is ai\b",
        r"\bi believe\b",
    ]
    for pat in drop_patterns:
        if re.search(pat, s, re.IGNORECASE):
            return True
    return False


def normalize_sentence_for_dedup(sent: str) -> str:
    s = sent.lower().strip()
    s = re.sub(r"[\"'`]", "", s)
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s


def clean_reasoning_text(text: str) -> str:
    sentences = split_sentences(text)

    cleaned = []
    seen = set()

    for sent in sentences:
        if should_drop_sentence(sent):
            continue

        norm = normalize_sentence_for_dedup(sent)
        if not norm:
            continue

        if norm in seen:
            continue

        seen.add(norm)
        cleaned.append(sent.strip())

    return " ".join(cleaned)


def extract_video_eval(text: str) -> dict:
    result = {
        "think": {},
        "answer": {}
    }

    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL | re.IGNORECASE)
    if think_match:
        think_content = think_match.group(1)

        think_pattern = re.compile(
            r"\[(Visual Quality|Motion\s*&\s*Physical Consistency|Prompt Alignment)\]\s*[:：]\s*(.*?)(?=\s*\[(?:Visual Quality|Motion\s*&\s*Physical Consistency|Prompt Alignment)\]\s*[:：]|$)",
            re.DOTALL | re.IGNORECASE
        )

        for key, value in think_pattern.findall(think_content):
            normalized_key = key.strip().lower()
            normalized_key = re.sub(r"\s*&\s*", " & ", normalized_key)
            normalized_key = re.sub(r"\s+", " ", normalized_key)
            result["think"][normalized_key] = value.strip()

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_content = answer_match.group(1)

        answer_pattern = re.compile(
            r"(Video Quality|Subject Movement|Physical Interaction|Cause-Effect|Subject Existence|Object Existence|Subject-Object Interaction)\s*[:：]\s*(Yes|No)\.?",
            re.IGNORECASE
        )

        for key, value in answer_pattern.findall(answer_content):
            normalized_key = key.strip().lower()
            normalized_key = re.sub(r"\s+", " ", normalized_key)
            result["answer"][normalized_key] = value.capitalize()

    return result


def extract_textual_prompt(text: str) -> Optional[str]:
    match = re.search(
        r"Textual prompt\s*:\s*(.*?)(?=\s*Assess whether the video is well-aligned)",
        text,
        re.DOTALL | re.IGNORECASE
    )
    return match.group(1).strip() if match else None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def title_case_label(label: str) -> str:
    mapping = {
        "video quality": "Video Quality",
        "subject movement": "Subject Movement",
        "physical interaction": "Physical Interaction",
        "cause-effect": "Cause-Effect",
        "subject existence": "Subject Existence",
        "object existence": "Object Existence",
        "subject-object interaction": "Subject-Object Interaction",
    }
    return mapping.get(label, label.title())


def format_group_reasoning(labels: List[str], shared_reason: str, answer: Dict[str, str]) -> List[str]:
    yes_labels = []
    no_labels = []

    for label in labels:
        val = answer.get(label, "")
        if val == "Yes":
            yes_labels.append(title_case_label(label))
        elif val == "No":
            no_labels.append(title_case_label(label))

    lines = []

    if yes_labels:
        lines.append(f"{', '.join(yes_labels)}: Good.")

    if no_labels:
        reason = clean_reasoning_text(shared_reason)
        if not reason:
            reason = "There are issues in this aspect."
        lines.append(f"{', '.join(no_labels)}: {reason}")

    return lines


def build_reasoning_trace(think: Dict[str, str], answer: Dict[str, str]) -> List[str]:
    visual_reason = clean_reasoning_text(think.get("visual quality", ""))
    motion_reason = clean_reasoning_text(think.get("motion & physical consistency", ""))
    prompt_reason = clean_reasoning_text(think.get("prompt alignment", ""))

    if not visual_reason and answer.get("video quality", "") == "No":
        visual_reason = "The video has noticeable visual defects."

    if not motion_reason and any(
        answer.get(k, "") == "No"
        for k in ["subject movement", "physical interaction", "cause-effect"]
    ):
        motion_reason = "There are issues with motion or physical consistency."

    if not prompt_reason and any(
        answer.get(k, "") == "No"
        for k in ["subject existence", "object existence", "subject-object interaction"]
    ):
        prompt_reason = "The video is not fully aligned with the prompt."

    lines = []

    # Video Quality 单独处理
    if answer.get("video quality", "") == "Yes":
        lines.append("Video Quality: Good.")
    elif answer.get("video quality", "") == "No":
        lines.append(f"Video Quality: {visual_reason}")

    # Motion & Physical Consistency 团
    lines.extend(format_group_reasoning(
        ["subject movement", "physical interaction", "cause-effect"],
        motion_reason,
        answer
    ))

    # Prompt Alignment 团
    lines.extend(format_group_reasoning(
        ["subject existence", "object existence", "subject-object interaction"],
        prompt_reason,
        answer
    ))

    return lines


def build_answer_dict(answer: Dict[str, str]) -> Dict[str, str]:
    fields = {
        "video_quality": answer.get("video quality", ""),
        "subject_movement": answer.get("subject movement", ""),
        "physical_interaction": answer.get("physical interaction", ""),
        "cause_effect": answer.get("cause-effect", ""),
        "subject_existence": answer.get("subject existence", ""),
        "object_existence": answer.get("object existence", ""),
        "subject_object_interaction": answer.get("subject-object interaction", ""),
    }
    return fields


def convert_item(item: Dict[str, Any]) -> Dict[str, Any]:
    human = item["conversations"][0]["value"]
    gpt = item["conversations"][1]["value"]

    prompt = extract_textual_prompt(human)
    if prompt:
        prompt = prompt.replace("..", ".")

    new_human = question.format(prompt=prompt if prompt else "")

    ret = extract_video_eval(gpt)
    think = ret["think"]
    answer = ret["answer"]

    think_lines = build_reasoning_trace(think, answer)
    answer_replace = build_answer_dict(answer)

    new_think = "<think>\n" + "\n".join(think_lines) + "\n</think>"

    new_answer = (
        "\n<answer>\n"
        "Video Quality: {video_quality}. "
        "Subject Movement: {subject_movement}. "
        "Physical Interaction: {physical_interaction}. "
        "Cause-Effect: {cause_effect}. "
        "Subject Existence: {subject_existence}. "
        "Object Existence: {object_existence}. "
        "Subject-Object Interaction: {subject_object_interaction}.\n"
        "</answer>"
    ).format(**answer_replace)

    new_item = {
        "videos": item["videos"],
        "conversations": [
            {
                "from": "human",
                "value": "<video>\n" + new_human
            },
            {
                "from": "gpt",
                "value": new_think + new_answer
            }
        ]
    }
    return new_item


def main():
    input_file = "data/train_t.json"
    output_file = "data/train_t_polished_v3.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    skipped = 0

    for item in data:
        try:
            new_item = convert_item(item)
            new_data.append(new_item)
        except Exception as e:
            skipped += 1
            print(f"Skip one item due to error: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(new_data)} items to {output_file}. Skipped {skipped} items.")


if __name__ == "__main__":
    main()