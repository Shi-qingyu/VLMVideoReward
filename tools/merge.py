import json
import re
import argparse
from pathlib import Path


NEW_DIMENSIONS = [
    "Video Quality",
    "Motion & Interaction",
    "Prompt Alignment",
]


MERGE_GROUPS = {
    "Motion & Interaction": [
        "Subject Movement",
        "Physical Interaction",
        "Cause-Effect",
    ],
    "Prompt Alignment": [
        "Subject Existence",
        "Object Existence",
        "Subject-Object Interaction",
    ],
}


OLD_TO_NEW_DIM = {
    "Video Quality": "Video Quality",

    "Subject Movement": "Motion & Interaction",
    "Physical Interaction": "Motion & Interaction",
    "Cause-Effect": "Motion & Interaction",

    "Subject Existence": "Prompt Alignment",
    "Object Existence": "Prompt Alignment",
    "Subject-Object Interaction": "Prompt Alignment",
}


NEW_INSTRUCTION = """Evaluate the video according to the following dimensions.
Video Quality: whether the video is free from major visual defects, including blur, lack of detail, poor texture, lighting issues, color distortion, flickering, and overexposure.
Motion & Interaction: whether the subject's motion is natural, smooth, and realistic; whether interactions among subjects and/or objects are physically plausible; and whether causal relationships are correctly depicted.
Prompt Alignment: whether the subject and object described in the prompt appear accurately, and whether the subject-object interaction described in the prompt is correctly represented."""


def merge_yes_no(values):
    """
    Merge Yes/No labels.

    Rule:
    - all Yes -> Yes
    - otherwise -> No
    """
    return "Yes" if all(v == "Yes" for v in values) else "No"


def parse_answer(answer_text):
    """
    Parse answer block like:

    Video Quality: Yes. Subject Movement: No. ...

    into:

    {
        "Video Quality": "Yes",
        "Subject Movement": "No",
        ...
    }
    """
    result = {}

    pattern = r"([A-Za-z&\- ]+):\s*(Yes|No)"
    for dim, label in re.findall(pattern, answer_text):
        result[dim.strip()] = label.strip()

    return result


def extract_answer_block(text):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.S)
    if not match:
        return None
    return match.group(1).strip()


def extract_think_block(text):
    match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.S)
    if not match:
        return None
    return match.group(1).strip()


def rebuild_answer_dict(old_answer_dict):
    """
    Build merged answer dict.
    """
    new_answer = {}

    new_answer["Video Quality"] = old_answer_dict.get("Video Quality", "No")

    for new_dim, old_dims in MERGE_GROUPS.items():
        old_values = [old_answer_dict.get(dim, "No") for dim in old_dims]
        new_answer[new_dim] = merge_yes_no(old_values)

    return new_answer


def format_answer(new_answer_dict):
    """
    Format merged answer dict into one-line answer.
    """
    return " ".join(
        f"{dim}: {new_answer_dict[dim]}."
        for dim in NEW_DIMENSIONS
    )


def remove_old_heading(line):
    """
    Remove old dimension heading.

    Examples:
    - "Subject Movement: Good." -> "Good."
    - "Physical Interaction, Cause-Effect: bad." -> "bad."
    """
    return re.sub(
        r"^([A-Za-z&\- ]+(?:,\s*[A-Za-z&\- ]+)*):\s*",
        "",
        line.strip()
    ).strip()


def ensure_sentence_ending(text):
    """
    Ensure a reasoning fragment ends with punctuation.

    If the text already ends with English or Chinese punctuation, keep it.
    """
    text = text.strip()
    if not text:
        return text

    if text[-1] not in ".!?。！？":
        text += "."

    return text


def strip_good_when_negative(reason_text, final_label):
    """
    If merged dimension is No, remove conflicting standalone positive verdicts.

    Examples:
    - "Good. <t>0.0s-5.0s</t> The movement is stiff."
      -> "<t>0.0s-5.0s</t> The movement is stiff."

    - "Good"
      -> ""
    """
    reason_text = reason_text.strip()

    if final_label != "No":
        return reason_text

    # Remove standalone "Good." or "Good" at the beginning.
    reason_text = re.sub(
        r"^\s*Good\.?\s*",
        "",
        reason_text,
        flags=re.I
    )

    # Remove repeated standalone "Good." after spaces.
    # This avoids cases like: "Good. Good. <t>..."
    reason_text = re.sub(
        r"(?<=\s)Good\.?\s*",
        "",
        reason_text,
        flags=re.I
    )

    # Clean spaces.
    reason_text = re.sub(r"\s+", " ", reason_text).strip()

    return reason_text


def build_merged_reasoning(lines, final_label):
    """
    Merge reasoning lines for one new dimension.

    Steps:
    1. remove old headings
    2. if final label is No, remove conflicting Good
    3. ensure each fragment ends with punctuation
    4. join fragments
    """
    fragments = []

    for line in lines:
        fragment = remove_old_heading(line)
        fragment = strip_good_when_negative(fragment, final_label)
        fragment = ensure_sentence_ending(fragment)

        if fragment:
            fragments.append(fragment)

    merged = " ".join(fragments)
    merged = re.sub(r"\s+", " ", merged).strip()

    return merged


def detect_line_dimension(line):
    """
    Detect which old dimension a reasoning line belongs to.

    Supports lines such as:
    - "Subject Movement: ..."
    - "Physical Interaction, Cause-Effect: ..."
    - "Subject Existence, Object Existence: ..."
    """
    prefix_match = re.match(r"^([A-Za-z&\- ]+(?:,\s*[A-Za-z&\- ]+)*):", line)
    if not prefix_match:
        return None

    prefix = prefix_match.group(1)
    old_dims = [x.strip() for x in prefix.split(",")]

    new_dims = []
    for old_dim in old_dims:
        if old_dim in OLD_TO_NEW_DIM:
            new_dims.append(OLD_TO_NEW_DIM[old_dim])

    if not new_dims:
        return None

    # If multiple old dims map to the same new dim, return that.
    if len(set(new_dims)) == 1:
        return new_dims[0]

    # Rare case: one line mentions dims that map to different new dims.
    # Keep it in the first mapped dimension.
    return new_dims[0]


def simplify_think_block(text, new_answer_dict):
    """
    Rewrite <think> block using merged dimensions.
    """
    think = extract_think_block(text)
    if think is None:
        return None

    lines = [line.strip() for line in think.splitlines() if line.strip()]

    grouped_lines = {
        "Video Quality": [],
        "Motion & Interaction": [],
        "Prompt Alignment": [],
    }

    for line in lines:
        new_dim = detect_line_dimension(line)

        if new_dim is None:
            # Fallback:
            # If no heading is detected, put it into Motion & Interaction.
            # This avoids dropping information.
            new_dim = "Motion & Interaction"

        grouped_lines[new_dim].append(line)

    new_lines = []

    for dim in NEW_DIMENSIONS:
        lines_for_dim = grouped_lines[dim]
        if not lines_for_dim:
            continue

        reason = build_merged_reasoning(
            lines_for_dim,
            new_answer_dict[dim]
        )

        if reason:
            new_lines.append(f"{dim}: {reason}")

    return "\n".join(new_lines)


def update_human_prompt(value):
    """
    Replace old dimension definitions in the human prompt.
    Keep the original Prompt: ... part unchanged.
    """
    prompt_idx = value.find("Prompt:")
    if prompt_idx == -1:
        return value

    prefix = value[:prompt_idx]
    suffix = value[prompt_idx:]

    intro_match = re.search(
        r"^(<video>\nSuppose you are an expert.*?\n)",
        prefix,
        flags=re.S
    )

    if intro_match:
        intro = intro_match.group(1)
        return intro + NEW_INSTRUCTION + "\n" + suffix

    return "<video>\n" + NEW_INSTRUCTION + "\n" + suffix


def update_gpt_response(value):
    """
    Update assistant response:
    - merge answer labels
    - merge think reasoning
    """
    answer_block = extract_answer_block(value)
    if answer_block is None:
        return value

    old_answer_dict = parse_answer(answer_block)
    new_answer_dict = rebuild_answer_dict(old_answer_dict)
    new_answer = format_answer(new_answer_dict)

    new_think = simplify_think_block(value, new_answer_dict)

    if new_think is None:
        return f"<answer>\n{new_answer}\n</answer>"

    return f"<think>\n{new_think}\n</think>\n<answer>\n{new_answer}\n</answer>"


def convert_item(item):
    conversations = item.get("conversations", [])

    for msg in conversations:
        role = msg.get("from")
        value = msg.get("value", "")

        if role == "human":
            msg["value"] = update_human_prompt(value)

        elif role == "gpt":
            msg["value"] = update_gpt_response(value)

    return item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        required=True,
        help="Input json file"
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output json file"
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=2,
        help="JSON indent. Use -1 for compact output."
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON should be a list.")

    new_data = [convert_item(item) for item in data]

    with output_path.open("w", encoding="utf-8") as f:
        if args.indent == -1:
            json.dump(new_data, f, ensure_ascii=False)
        else:
            json.dump(new_data, f, ensure_ascii=False, indent=args.indent)

    print(f"Saved converted file to: {output_path}")


if __name__ == "__main__":
    main()