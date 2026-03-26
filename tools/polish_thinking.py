import json
import copy
import time
import re
import os
from tqdm import tqdm
from openai import AzureOpenAI
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# =========================
# Azure OpenAI Config
# =========================
client = AzureOpenAI(
    azure_endpoint="https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl",
    api_version="2025-04-01-preview",
    api_key="5iyrOGLAr5Xscz6WqS3Zv8ePc5as5cKL",
    timeout=111.0
)

MODEL_NAME = "gpt-5-2025-08-07"


# =========================
# Qwen3 Config
# =========================
QWEN_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
qwen_tokenizer = None
qwen_model = None


def load_qwen_model():
    global qwen_tokenizer, qwen_model
    if qwen_tokenizer is None or qwen_model is None:
        print(f"Loading Qwen model: {QWEN_MODEL_NAME}")
        qwen_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL_NAME)
        qwen_model = AutoModelForCausalLM.from_pretrained(
            QWEN_MODEL_NAME,
            torch_dtype="auto",
            device_map="auto"
        )
    return qwen_tokenizer, qwen_model


SYSTEM_PROMPT = """
You are a text polishing assistant.

Your task is to improve the fluency and clarity of the text inside the <think> section ONLY, while strictly preserving:
1. The original structure and format (e.g., [Visual Quality], [Motion & Physical Consistency], [Prompt Alignment])
2. The meaning of each statement

Rules:
- Rewrite each item to be natural, fluent, and grammatically correct English sentence
- Do NOT modify the <answer> section in any way

My input:
{user_prompt}
"""


def build_prompt(user_prompt: str) -> str:
    return SYSTEM_PROMPT.format(user_prompt=user_prompt)


def polish_with_gpt(user_prompt: str) -> str:
    full_prompt = build_prompt(user_prompt)
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "user", "content": full_prompt}
            ],
        )

        msg = response.choices[0].message
        print("finish_reason:", response.choices[0].finish_reason)
        print("usage:", response.usage)

        content = (msg.content or "").strip()
        return content if content else user_prompt

    except Exception as e:
        print(f"Azure API failed: {e}")
        return user_prompt


def polish_with_qwen(user_prompt: str, max_new_tokens: int = 4096) -> str:
    full_prompt = build_prompt(user_prompt)

    try:
        tokenizer, model = load_qwen_model()

        messages = [
            {"role": "user", "content": full_prompt}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
        content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return content if content else user_prompt

    except Exception as e:
        print(f"Qwen polish failed: {e}")
        return user_prompt


def polish_text(user_prompt: str, backend: str = "azure") -> str:
    if backend == "qwen":
        return polish_with_qwen(user_prompt)
    return polish_with_gpt(user_prompt)


def save_json(data, output_path):
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    os.replace(tmp_path, output_path)


if __name__ == "__main__":
    BACKEND = "azure"

    input_path = "data/eval_filtered.json"
    output_path = "data/eval_polished.json"

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

            # polished_text = polish_text(ori_text, backend=BACKEND)
            polished_text = fix_subject_object_interaction(ori_text)

            polished_text = (
                polished_text
                .replace(".</think>", ".\n</think>")
                .replace("[Prompt Alignment]: No.", "[Prompt Alignment]:")
            )
            polished_item = copy.deepcopy(item)
            polished_item["conversations"][-1]["value"] = polished_text
            polished_data.append(polished_item)

            save_json(polished_data, output_path)

    except KeyboardInterrupt:
        print("\nDetected KeyboardInterrupt, saving current progress...")
        save_json(polished_data, output_path)
        print(f"Progress saved to {output_path}, total saved: {len(polished_data)}")
        raise

    except Exception as e:
        print(f"\nError occurred: {e}")
        print("Saving current progress...")
        save_json(polished_data, output_path)
        print(f"Progress saved to {output_path}, total saved: {len(polished_data)}")
        raise

    finally:
        save_json(polished_data, output_path)
        print(f"Final progress saved to {output_path}, total saved: {len(polished_data)}")