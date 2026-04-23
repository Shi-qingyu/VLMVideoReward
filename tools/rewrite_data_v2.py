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
    # 按英文句号/问号/感叹号切句，保留标点
    parts = re.split(r'(?<=[.!?])\s+', text)
    return [p.strip() for p in parts if p.strip()]


def should_drop_sentence(sent: str) -> bool:
    s = sent.lower().strip()

    # 去掉提到 AI / AI-generated / discrepancy 的元话语
    drop_patterns = [
        r"\bai-generated\b",
        r"\bai generated\b",
        r"\bi believe the video is ai\b",
        r"\bI believe\b".lower(),   # 泛化一点
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


def join_sentences(parts: list[str]) -> str:
    seen = set()
    out = []
    for p in parts:
        p = normalize_space(p)
        if not p:
            continue
        if p not in seen:
            seen.add(p)
            out.append(p)
    return " ".join(out)


def build_reasoning_trace(think, answer):
    visual_reason = clean_reasoning_text(think.get("visual quality", ""))
    motion_reason = clean_reasoning_text(think.get("motion & physical consistency", ""))
    prompt_reason = clean_reasoning_text(think.get("prompt alignment", ""))
    if "bad" in prompt_reason.lower() or " ai" in prompt_reason.lower():
        prompt_reason = "The subject-object interaction is not well aligned with the prompt."

    vq = answer.get("video quality", "")
    sm = answer.get("subject movement", "")
    pi = answer.get("physical interaction", "")
    ce = answer.get("cause-effect", "")
    se = answer.get("subject existence", "")
    oe = answer.get("object existence", "")
    soi = answer.get("subject-object interaction", "")

    if vq == "Yes":
        visual_out = "The video quality is good."
    else:
        visual_out = visual_reason or "The video has noticeable visual defects."

    motion_group = [sm, pi, ce]
    if all(x == "Yes" for x in motion_group):
        motion_out = "The subject movement, physical interaction, and cause-effect are all reasonable."
    else:
        motion_out = motion_reason or "There are issues with motion, physical interaction, or cause-effect consistency."

    prompt_group = [se, oe, soi]
    if all(x == "Yes" for x in prompt_group):
        prompt_out = "The video is well aligned with the prompt."
    else:
        prompt_out = prompt_reason or "The video is not fully aligned with the prompt."

    prompt_sents = split_sentences(prompt_out)
    motion_norms = {normalize_sentence_for_dedup(s) for s in split_sentences(motion_out)}

    filtered_prompt_sents = []
    for s in prompt_sents:
        if normalize_sentence_for_dedup(s) not in motion_norms:
            filtered_prompt_sents.append(s)

    prompt_out = " ".join(filtered_prompt_sents).strip()
    if not prompt_out:
        prompt_out = "The video is not fully aligned with the prompt."

    return {
        "visual_quality": visual_out,
        "subject_movement_physical_interaction_cause_effect": motion_out,
        "subject_existence_object_existence_subject_object_interaction": prompt_out,
    }


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
    prompt = prompt.replace("..", ".") # small fix
    new_human = question.format(prompt=prompt if prompt else "")

    ret = extract_video_eval(gpt)
    think = ret["think"]
    answer = ret["answer"]

    think_replace = build_reasoning_trace(think, answer)
    answer_replace = build_answer_dict(answer)

    new_think = (
        "<think>\n"
        "{visual_quality} "
        "{subject_movement_physical_interaction_cause_effect} "
        "{subject_existence_object_existence_subject_object_interaction}\n"
        "</think>"
    ).format(**think_replace)

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
    output_file = "data/train_t_polished_v2.json"

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []
    for item in data:
        try:
            new_item = convert_item(item)
            new_data.append(new_item)
        except Exception as e:
            print(f"Skip one item due to error: {e}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=2)

    print(f"Done. Saved {len(new_data)} items to {output_file}")


if __name__ == "__main__":
    main()