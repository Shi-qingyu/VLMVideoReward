import json
import os
import re
import argparse
import hashlib
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


template = """
**Answer:** {answer}
**Ground Truth:** {ground_truth}
First, determine whether the sentiments expressed in the two evaluations are consistent. Then, further assess whether the reasons identified for those issues are also aligned.
Based on the above criteria, assign a consistency score ranging from 0 to 1, where 0 indicates complete inconsistency and 1 indicates complete consistency. Please provide your judgment along with a clear explanation of your reasoning.
"""


SCORE_PATTERN = re.compile(
    r"Consistency\s*Score\s*[:：]?\s*\**\s*([01](?:\.\d+)?)",
    re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Judge with pure vLLM tensor parallelism")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen3-30B-A3B-Instruct-2507",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_model_len",
        type=int,
        default=8192,
    )
    return parser.parse_args()


def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def get_sample_key(item):
    raw = json.dumps(
        {
            "answer": item.get("answer", ""),
            "ground_truth": item.get("ground_truth", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def extract_consistency_score(judgment: str):
    if not judgment:
        return None

    match = SCORE_PATTERN.search(judgment)
    if not match:
        return None

    try:
        score = float(match.group(1))
    except ValueError:
        return None

    if 0.0 <= score <= 1.0:
        return score
    return None


def calculate_consistency_score(judgments):
    overall_score = 0.0
    valid_scores = 0

    for item in judgments:
        judgment = item.get("judgment", "")
        score = extract_consistency_score(judgment)
        if score is not None:
            overall_score += score
            valid_scores += 1

    if valid_scores > 0:
        return overall_score / valid_scores
    return None


def load_existing_results(output_file):
    if not os.path.exists(output_file):
        return [], set()

    with open(output_file, "r", encoding="utf-8") as f:
        existing_results = json.load(f)

    done_keys = set()
    for item in existing_results:
        key = get_sample_key(item)
        done_keys.add(key)

    return existing_results, done_keys


def atomic_save_json(data, output_file):
    tmp_file = output_file + ".tmp"
    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_file, output_file)


def main():
    args = parse_args()

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        name, ext = os.path.splitext(base_name)
        args.output_file = os.path.join(
            os.path.dirname(args.input_file),
            f"{name}_judgment{ext}",
        )

    existing_results, done_keys = load_existing_results(args.output_file)

    pending_data = []
    for item in data:
        key = get_sample_key(item)
        if key not in done_keys:
            pending_data.append(item)

    if not pending_data:
        score = calculate_consistency_score(existing_results)
        print(f"all samples already processed, consistency score: {score}")
        return

    print(
        f"resume enabled: total={len(data)}, done={len(existing_results)}, "
        f"pending={len(pending_data)}"
    )

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

    results = list(existing_results)
    total_batches = (len(pending_data) + args.batch_size - 1) // args.batch_size

    for batch in tqdm(
        batched(pending_data, args.batch_size),
        total=total_batches,
        desc="judge",
    ):
        prompts = []
        batch_items = []

        for item in batch:
            answer = item["answer"]
            ground_truth = item["ground_truth"]

            input_text = template.format(
                answer=answer,
                ground_truth=ground_truth,
            )

            messages = [{"role": "user", "content": input_text}]
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            prompts.append(text)
            batch_items.append(item)

        outputs = llm.generate(prompts, sampling_params)

        for item, output in zip(batch_items, outputs):
            judgment = output.outputs[0].text.strip()
            results.append(
                {
                    "answer": item["answer"],
                    "ground_truth": item["ground_truth"],
                    "judgment": judgment,
                }
            )

        atomic_save_json(results, args.output_file)

    score = calculate_consistency_score(results)
    print(f"consistency score: {score}")


if __name__ == "__main__":
    main()