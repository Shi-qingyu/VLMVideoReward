import os
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

import re
import json
import argparse
import torch
from tqdm import tqdm
from collections import defaultdict
from torch.utils.data import DataLoader
import multiprocessing as mp

from qwenvl.dataset.data_processor import make_rl_data_module
from transformers import AutoProcessor
from vllm import LLM, SamplingParams


OLD_KEYS = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--dataset_use", type=str, default="videoreward_eval_polished_v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--using_cot", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")

    # vLLM args
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    return parser.parse_args()


def normalize_label(text: str) -> str:
    if text is None:
        return "fail"
    text = str(text).strip().lower()
    text = re.sub(r"[^\w]", "", text)
    if text in ["yes", "y", "true", "1", "good"]:
        return "yes"
    if text in ["no", "n", "false", "0", "bad"]:
        return "no"
    return "fail"


def parse_key_value_pairs(text: str, keys):
    parsed = {}
    for key in keys:
        pattern = rf"{re.escape(key)}\s*:\s*([A-Za-z]+)"
        match = re.search(pattern, text, re.I)
        parsed[key] = normalize_label(match.group(1)) if match else "fail"
    return parsed


def parse_output(text: str):
    answer_match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, re.S)
    body = answer_match.group(1).strip() if answer_match else text.strip()

    parsed_old = parse_key_value_pairs(body, OLD_KEYS)
    return parsed_old


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def calculate_metrics(outputs):
    stats = {key: defaultdict(int) for key in OLD_KEYS}

    for item in outputs:
        pred_dict = parse_output(item["answer"])
        gt_dict = parse_output(item["ground_truth"])

        for key in OLD_KEYS:
            p = pred_dict.get(key, "fail")
            g = gt_dict.get(key, "fail")

            if g not in ["yes", "no"]:
                continue

            s = stats[key]
            s["total"] += 1
            if g == "yes":
                s["gt_yes"] += 1
            else:
                s["gt_no"] += 1

            if p == g:
                s["correct"] += 1
                if g == "yes":
                    s["tp"] += 1
                else:
                    s["tn"] += 1
            else:
                if p == "yes":
                    s["fp"] += 1
                elif p == "no":
                    s["fn"] += 1
                else:
                    s["fail_pred"] += 1

    metrics = {}
    for key in OLD_KEYS:
        s = stats[key]
        acc = safe_div(s["correct"], s["total"])
        prec = safe_div(s["tp"], s["tp"] + s["fp"])
        rec = safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = safe_div(2 * prec * rec, prec + rec)
        metrics[key] = {
            **s,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
        }

    all_fields = ["tp", "tn", "fp", "fn", "correct", "total", "gt_yes", "gt_no", "fail_pred"]
    summary = {f: sum(metrics[k].get(f, 0) for k in OLD_KEYS) for f in all_fields}
    summary["accuracy"] = safe_div(summary["correct"], summary["total"])
    summary["precision"] = safe_div(summary["tp"], summary["tp"] + summary["fp"])
    summary["recall"] = safe_div(summary["tp"], summary["tp"] + summary["fn"])
    summary["f1"] = safe_div(
        2 * summary["precision"] * summary["recall"],
        summary["precision"] + summary["recall"],
    )

    return metrics, summary


def print_table(metrics, summary):
    header = ["Dimension", "GT(Y/N)", "TP", "TN", "FP", "FN", "Acc", "Prec", "Rec", "F1"]
    rows = []
    for k in OLD_KEYS:
        m = metrics[k]
        rows.append([
            k[:28],
            f"{m['gt_yes']}/{m['gt_no']}",
            m["tp"],
            m["tn"],
            m["fp"],
            m["fn"],
            f"{m['accuracy'] * 100:.1f}%",
            f"{m['precision'] * 100:.1f}%",
            f"{m['recall'] * 100:.1f}%",
            f"{m['f1'] * 100:.1f}%",
        ])

    s = summary
    rows.append([
        "OVERALL",
        f"{s['gt_yes']}/{s['gt_no']}",
        s["tp"],
        s["tn"],
        s["fp"],
        s["fn"],
        f"{s['accuracy'] * 100:.1f}%",
        f"{s['precision'] * 100:.1f}%",
        f"{s['recall'] * 100:.1f}%",
        f"{s['f1'] * 100:.1f}%",
    ])

    col_w = [max(len(str(r[i])) for r in rows + [header]) for i in range(len(header))]
    sep = "+" + "+".join("-" * (w + 2) for w in col_w) + "+"

    print(f"\n{sep}\n| " + " | ".join(str(header[i]).ljust(col_w[i]) for i in range(len(header))) + f" |\n{sep}")
    for i, r in enumerate(rows):
        print("| " + " | ".join(str(r[j]).ljust(col_w[j]) for j in range(len(r))) + " |")
        if i == len(rows) - 2:
            print(sep.replace("-", "="))
        else:
            print(sep)


def to_vllm_chat_format(sample):
    msg = sample[0]
    text_parts = []
    video_path = None

    for part in msg["content"]:
        if part.get("type") == "video":
            video_path = part["video"]
        elif part.get("type") == "text":
            text_parts.append(part["text"])

    if video_path is None:
        raise ValueError("No video found in prompt")

    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {
                        "url": f"file://{video_path}"
                    }
                },
                {
                    "type": "text",
                    "text": "\n".join(text_parts),
                },
            ],
        }
    ]


@torch.inference_mode()
def run_eval(args):
    model_name = "-".join(args.model_path.split("/")[1:])
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")

    results = []
    if os.path.exists(output_path):
        print(f"Loading cached results from {output_path}")
        with open(output_path, "r", encoding="utf-8") as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                print("Warning: JSON file corrupted or empty. Starting from scratch.")
                results = []
        metrics, summary = calculate_metrics(results)
        print_table(metrics, summary)
        return

    llm = LLM(
        model=args.model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path="/mnt/bn/xiangtai-training-data-video/sqy/projects/videorewardmodel/data/videos/",
    )
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    processor = AutoProcessor.from_pretrained(args.model_path)
    data_module = make_rl_data_module(processor=processor, data_args=args)
    dataloader = DataLoader(
        data_module["train_dataset"],
        collate_fn=data_module["data_collator"],
        batch_size=args.batch_size,
    )

    results = []
    for batch in tqdm(dataloader, desc="Inference"):
        user_prompts = batch["user"]
        gt_prompts = batch["gt"]

        converted_prompts = [to_vllm_chat_format(p) for p in user_prompts]
        outputs = llm.chat(converted_prompts, sampling_params=sampling_params)
        preds = [o.outputs[0].text for o in outputs]

        for i in range(len(preds)):
            video = user_prompts[i][0]["content"][0]["video"]
            gt_text = gt_prompts[i][0]["content"][0]["text"]
            item = {
                "video": video,
                "answer": preds[i],
                "ground_truth": gt_text,
            }
            results.append(item)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)

    metrics, summary = calculate_metrics(results)
    print_table(metrics, summary)

    metrics_path = output_path.replace(".json", "_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"per_dim": metrics, "overall": summary}, f, indent=4, ensure_ascii=False)

    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_eval(get_args())