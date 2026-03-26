import re
import os
import json
import copy

from tqdm import tqdm


def save_json(data, output_path):
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, output_path)


def fix_subject_object_interaction(text: str) -> str:
    soi_pattern = r"(Subject-Object Interaction:\s*)(.*?)(?=\n|</answer>)"
    soi_match = re.search(soi_pattern, text, re.DOTALL)
    if not soi_match:
        return text

    value = soi_match.group(2).strip().replace("..", ".").replace(". .", ".")

    if value in {"Yes.", "No."}:
        return text

    pa_pattern = r"(\[Prompt Alignment\]:\s*)(.*?)(?=\s*\[.*?\]:|</think>)"
    pa_match = re.search(pa_pattern, text, re.DOTALL)
    if pa_match:
        pa_prefix = pa_match.group(1)
        pa_value = pa_match.group(2).strip()

        if pa_value.endswith("."):
            new_pa_value = f"{pa_value} {value}"
        else:
            new_pa_value = f"{pa_value}. {value}"

        text = re.sub(
            pa_pattern,
            f"{pa_prefix}{new_pa_value}",
            text,
            count=1,
            flags=re.DOTALL
        )

    text = re.sub(
        soi_pattern,
        "Subject-Object Interaction: No.",
        text,
        count=1,
        flags=re.DOTALL
    )

    return text


if __name__ == "__main__":
    BACKEND = "azure"

    input_path = "data/train_polished.json"
    output_path = "data/train_fixed.json"

    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            polished_data = json.load(f)
        start_idx = len(polished_data)
        print(f"Resume from index {start_idx}, already polished {start_idx} items.")
    else:
        polished_data = []
        start_idx = 0

    try:
        for i in tqdm(range(start_idx, len(data)), desc=f"Polishing with {BACKEND}"):
            item = data[i]
            ori_text = item["conversations"][-1]["value"]
            polished_text = fix_subject_object_interaction(ori_text)

            polished_text = (
                polished_text
                .replace(".</think>", ".\n</think>")
                .replace("[Prompt Alignment]: No.", "[Prompt Alignment]:")
            )
            polished_item = copy.deepcopy(item)
            polished_item["conversations"][-1]["value"] = polished_text
            polished_data.append(polished_item)

    finally:
        save_json(polished_data, output_path)
        print(f"Final progress saved to {output_path}, total saved: {len(polished_data)}")