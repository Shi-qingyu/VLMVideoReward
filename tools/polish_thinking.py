import json
import copy
import time
import re
from tqdm import tqdm
from openai import AzureOpenAI

client = AzureOpenAI(
    azure_endpoint="https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl",
    api_version="2025-04-01-preview",
    api_key="5iyrOGLAr5Xscz6WqS3Zv8ePc5as5cKL",
    timeout=111.0
)

MODEL_NAME = "gpt-5-2025-08-07"


SYSTEM_PROMPT = """
You are a text polishing assistant.

Your task is to improve the fluency and clarity of the text inside the <think> section ONLY, while strictly preserving:
1. The original structure and format
2. All bracketed labels (e.g., [Visual Quality], [Motion & Physical Consistency], [Prompt Alignment])
3. The number of items and their order
4. The meaning of each statement

Rules:
- Rewrite each item to be natural, fluent, and grammatically correct English
- Keep each item in the format: [Label]: sentence.
- Do NOT merge, split, or reorder items
- Do NOT modify the <answer> section in any way
- Do NOT add or remove any information
- Avoid redundant or awkward phring like "unable to identify gender" → make it smoother but same meaning

Input:
<think>
[Visual Quality]: Garbage Symbols. [Motion & Physical Consistency]: Good. [Prompt Alignment]: The subject's face is not visible, unable to identify gender.
</think>
<answer>
Video Quality: No. Subject Movement: Yes. Physical Interaction: Yes. Cause-Effect: Yes. Subject Existence: No. Object Existence: Yes. Subject-Object Interaction: Yes.
</answer>

Output:
<think>
[Visual Quality]: Contains visual artifacts or corrupted symbols. [Motion & Physical Consistency]: The motion is physically consistent. [Prompt Alignment]: The subject's face is not visible, making it impossible to determine gender.
</think>
<answer>
Video Quality: No. Subject Movement: Yes. Physical Interaction: Yes. Cause-Effect: Yes. Subject Existence: No. Object Existence: Yes. Subject-Object Interaction: Yes.
</answer>
"""


def polish_with_llm(user_prompt):
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt}
            ],
        )

        msg = response.choices[0].message
        print("finish_reason:", response.choices[0].finish_reason)
        print("usage:", response.usage)

        content = (msg.content or "").strip()
        return content if content else user_prompt

    except Exception as e:
        print(f"API failed: {e}")
        return user_prompt
    

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
    with open("data/train_filtered.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    polished_data = []
    for item in tqdm(data, desc="Polishing with LLM", total=len(data)):
        ori_text = item["conversations"][-1]["value"]
        # polished_text = polish_with_llm(ori_text)
        polished_text = fix_subject_object_interaction(ori_text)
        if polished_text != ori_text:
            polished_text = polished_text.replace("[Prompt Alignment]: Yes", "[Prompt Alignment]: Bad")
            # print(polished_text)
            # print(ori_text)
        polished_item = copy.deepcopy(item)
        polished_item["conversations"][-1]["value"] = polished_text
        polished_data.append(polished_item)
    
    with open("data/train_polished.json", "w", encoding="utf-8") as f:
        json.dump(polished_data, f, ensure_ascii=False, indent=4)