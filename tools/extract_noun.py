import os
import json
import argparse
import hashlib
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


template = "From the following judgment, extract only one main visible anomalous subject or object. Choose the most salient concrete, visually identifiable noun mentioned in the judgment, using the exact words from the judgment. Ignore attributes, actions, abstract concepts, evaluations, and nationality/ethnicity descriptions. Judgment: {judgement}. Return only one subject or object."


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="data/train_fixed.json")
    parser.add_argument("--output_json", type=str, default="data/train_nouns.json")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--max_model_len", type=int, default=8192)
    return parser.parse_args()


def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def get_sample_key(d):
    if "id" in d:
        return str(d["id"])
    raw = json.dumps(
        {
            "video": d["videos"][0] if d.get("videos") else "",
            "judgement": d["conversations"][-1]["value"] if d.get("conversations") else "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def extract_thinking(judgement: str) -> str:
    start_pos = judgement.find("<think>")
    end_pos = judgement.find("</think>")
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        return judgement[start_pos + len("<think>"): end_pos]
    return judgement


def extract_answering(judgement: str) -> str:
    start_pos = judgement.find("<answer>")
    end_pos = judgement.find("</answer>")
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        return judgement[start_pos + len("<answer>"): end_pos]
    return judgement


def normalize_nouns(content: str):
    texts = [x.strip() for x in content.split(",") if x.strip()]
    texts = list(dict.fromkeys(texts))
    return texts


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

    for batch in tqdm(batched(data, args.batch_size), total=total_batches, desc="extract_nouns"):
        prompts = []
        batch_meta = []

        for d in batch:
            judgement = d["conversations"][-1]["value"]
            thinking = extract_thinking(judgement)
            answer = extract_answering(judgement)

            meta = {
                "key": get_sample_key(d),
                "video": d["videos"][0] if d.get("videos") else None,
                "judgement": judgement,
                "thinking": thinking,
                "raw_sample": d,
            }

            if "NO" not in answer.upper():
                results.append(
                    {
                        "key": meta["key"],
                        "video": meta["video"],
                        "thinking": meta["thinking"],
                        "nouns": [],
                        "sample": meta["raw_sample"],
                    }
                )
                continue

            prompt = template.format(judgement=thinking)
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(text)
            batch_meta.append(meta)

        if prompts:
            outputs = llm.generate(prompts, sampling_params)

            for meta, output in zip(batch_meta, outputs):
                content = output.outputs[0].text.strip()
                nouns = normalize_nouns(content)

                results.append(
                    {
                        "key": meta["key"],
                        "video": meta["video"],
                        "thinking": meta["thinking"],
                        "nouns": nouns,
                        "sample": meta["raw_sample"],
                    }
                )

        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()