import json
import os
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


template = """
**Answer:** {answer}
**Ground Truth:** {ground_truth}
These are two evaluations of AI-generated videos. First, determine whether the sentiments expressed in the two evaluations are consistent (positive or negative) across three dimensions. If the sentiments are consistent, then further assess whether the identified problematic subjects or objects—and the reasons for those issues—are also aligned.
Based on the above criteria, assign a consistency score ranging from 0 to 1, where 0 indicates complete inconsistency and 1 indicates complete consistency. Please provide your judgment along with a clear explanation of your reasoning.
"""


def parse_args():
    parser = argparse.ArgumentParser(description="Qwen Judge with vLLM + torchrun")
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
    return parser.parse_args()


def batched(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def split_data(data, rank, world_size):
    return data[rank::world_size]


def get_rank_info():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    return local_rank, rank, world_size


def build_output_paths(args, rank):
    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        name, ext = os.path.splitext(base_name)
        rank_output = os.path.join(
            os.path.dirname(args.input_file),
            f"{name}_judgment_rank{rank}{ext}",
        )
        final_output = os.path.join(
            os.path.dirname(args.input_file),
            f"{name}_judgment{ext}",
        )
    else:
        base, ext = os.path.splitext(args.output_file)
        rank_output = f"{base}_rank{rank}{ext}"
        final_output = args.output_file

    return rank_output, final_output


def merge_outputs(args, world_size):
    merged = []

    if args.output_file is None:
        base_name = os.path.basename(args.input_file)
        name, ext = os.path.splitext(base_name)
        final_output = os.path.join(
            os.path.dirname(args.input_file),
            f"{name}_judgment{ext}",
        )
    else:
        final_output = args.output_file

    for rank in range(world_size):
        rank_output, _ = build_output_paths(args, rank)
        if os.path.exists(rank_output):
            with open(rank_output, "r", encoding="utf-8") as f:
                merged.extend(json.load(f))

    with open(final_output, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=4, ensure_ascii=False)


def main():
    args = parse_args()
    local_rank, rank, world_size = get_rank_info()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(local_rank)

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    shard = split_data(data, rank, world_size)
    rank_output, _ = build_output_paths(args, rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    results = []
    total_batches = (len(shard) + args.batch_size - 1) // args.batch_size

    for batch in tqdm(
        batched(shard, args.batch_size),
        total=total_batches,
        desc=f"rank {rank}",
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
            results.append({
                "answer": item["answer"],
                "ground_truth": item["ground_truth"],
                "judgment": judgment,
            })

        with open(rank_output, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()