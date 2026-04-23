import os
import json
import argparse
import copy
import re
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


template = (
    "Please polish the following sentence into fluent English without adding any new details: {sentence}"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="data/train_fixed.json")
    parser.add_argument("--output_json", type=str, default="data/train_polished.json")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser.parse_args()

def remove_ai_sentence(text: str) -> str:
    """
    Remove any sentence containing 'AI' from a string.
    """
    pattern = r'[^.!?]*\bAI\b[^.!?]*[.!?]?\s*'
    cleaned = re.sub(pattern, '', text, flags=re.IGNORECASE)
    return cleaned.strip()


def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def extract_thinking(judgement: str) -> str:
    start_pos = judgement.find("<think>")
    end_pos = judgement.find("</think>")
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        return judgement[start_pos + len("<think>"): end_pos].strip()
    return judgement.strip()


def extract_answering(judgement: str) -> str:
    start_pos = judgement.find("<answer>")
    end_pos = judgement.find("</answer>")
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        return judgement[start_pos + len("<answer>"): end_pos].strip()
    return ""


def normalize_cot(content: str) -> str:
    return content.strip()


def replace_tag_content(text: str, tag_name: str, new_content: str) -> str:
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"

    start_pos = text.find(start_tag)
    end_pos = text.find(end_tag)

    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        prefix = text[: start_pos + len(start_tag)]
        suffix = text[end_pos:]
        return f"{prefix}\n{new_content}\n{suffix}"

    return text


def parse_yes_no_fields(answer_text: str) -> dict:
    """
    Parse lines like:
    Video Quality: Yes. Subject Movement: No. ...
    into a dict.
    """
    fields = {}
    pattern = re.findall(r"([A-Za-z&\-\s]+):\s*(Yes|No)", answer_text, flags=re.IGNORECASE)
    for k, v in pattern:
        fields[k.strip()] = v.strip().capitalize()
    return fields


def merge_two_yes_no(v1: str, v2: str) -> str:
    return "Yes" if v1 == "Yes" and v2 == "Yes" else "No"


def rewrite_answer(answer_text: str) -> str:
    fields = parse_yes_no_fields(answer_text)

    assert len(fields) == 7, f"Expected 7 fields, but got {fields}"
    
    video_quality = fields.get("Video Quality", "Failed")
    motion_quality = fields.get("Subject Movement", "Failed")

    physical_interaction = fields.get("Physical Interaction", "Failed")
    cause_effect = fields.get("Cause-Effect", "Failed")

    subject_existence = fields.get("Subject Existence", "Failed")
    object_existence = fields.get("Object Existence", "Failed")

    overall_alignment = fields.get("Subject-Object Interaction", "Failed")

    new_answer = (
        f"Video Quality: {video_quality}. "
        f"Subject Movement: {motion_quality}. "
        f"Physical Interaction: {physical_interaction}. "
        f"Cause-Effect: {cause_effect}. "
        f"Subject Existence: {subject_existence}. "
        f"Object Existence: {object_existence}. "
        f"Subject-Object Interaction: {overall_alignment}."
    )
    return new_answer


def extract_textual_prompt(question: str) -> str:
    match = re.search(
        r"Textual prompt:\s*(.*?)\nAssess whether",
        question,
        flags=re.DOTALL,
    )
    if match:
        return match.group(1).strip()
    return ""


def main():
    args = parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    results = []
    total_batches = (len(data) + args.batch_size - 1) // args.batch_size

    for batch in tqdm(batched(data, args.batch_size), total=total_batches, desc="rewrite_cot"):
        prompts = []
        batch_meta = []

        for d in batch:
            judgement = d["conversations"][-1]["value"]
            thinking = extract_thinking(judgement)
            answer = extract_answering(judgement)

            prompt = template.format(sentence=thinking)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(text)
            batch_meta.append(
                {
                    "raw_sample": d,
                    "judgement": judgement,
                    "answer": answer,
                }
            )

        outputs = llm.generate(prompts, sampling_params)

        for meta, output in zip(batch_meta, outputs):
            content = output.outputs[0].text.strip()
            cot = normalize_cot(content)
            cot = remove_ai_sentence(cot)

            new_sample = copy.deepcopy(meta["raw_sample"])
            old_judgement = new_sample["conversations"][-1]["value"]

            # replace think
            new_judgement = replace_tag_content(old_judgement, "think", cot)

            # replace answer
            old_answer = extract_answering(new_judgement)
            new_answer = rewrite_answer(old_answer)
            new_judgement = replace_tag_content(new_judgement, "answer", new_answer)

            new_sample["conversations"][-1]["value"] = new_judgement
            results.append(new_sample)

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()