import argparse
import json
import os
import re
from collections import defaultdict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

from inference_common import (
    build_generation_kwargs,
    build_template_kwargs,
    configure_internvl_processor,
    infer_model_type,
    load_model,
    load_processor,
    normalize_molmo2_messages,
    prepare_processor,
    trim_repeated_response,
)
from src.dataset.data_processor import make_rl_data_module
from src.train.checkpoint_utils import prepare_inference_model_dir


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


MERGED_KEYS = [
    "Video Quality",
    "Motion & Interaction",
    "Prompt Alignment",
]

LEGACY_KEYS = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
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

KEY_ALIASES = {
    "Video Quality": ["Video Quality", "Visual Quality"],
    "Motion & Interaction": [
        "Motion & Interaction",
        "Motion and Interaction",
        "Motion & Physical Consistency",
        "Motion and Physical Consistency",
    ],
    "Prompt Alignment": ["Prompt Alignment"],
}


def add_common_eval_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--backend",
        choices=["auto", "vllm", "hf"],
        default="auto",
        help="auto uses each model-specific script's default backend.",
    )
    parser.add_argument("--dataset_use", type=str, default="videoreward_eval_polished_v3")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument(
        "--device_map",
        default=os.environ.get("DEVICE_MAP", "auto"),
        help="Device map for HF model loading.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=0)
    parser.add_argument("--using_cot", action="store_true", default=True)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument(
        "--metric_schema",
        choices=["auto", "merged", "legacy"],
        default="auto",
        help="Metric dimensions: auto/merged use the 3-dim schema; legacy uses the old 7-dim schema.",
    )

    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.7)
    parser.add_argument("--max_num_seqs", type=int, default=8)
    parser.add_argument(
        "--allowed_local_media_path",
        type=str,
        default="/mnt/bn/xiangtai-training-data-video/sqy/projects/videorewardmodel/data/videos/",
    )

    parser.add_argument("--video_max_frames", type=int, default=8)
    parser.add_argument("--video_fps", type=float, default=1.0)
    parser.add_argument("--internvl_image_size", type=int, default=448)
    parser.add_argument("--internvl_min_patches", type=int, default=1)
    parser.add_argument("--internvl_max_patches", type=int, default=4)
    parser.add_argument("--molmo2_image_size", type=int, default=378)
    parser.add_argument(
        "--molmo2_video_frame_sampling_mode",
        default="uniform_last_frame",
    )
    parser.add_argument("--attn_implementation", default=os.environ.get("ATTN_IMPLEMENTATION"))
    return parser


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


def metric_keys_from_schema(schema: str):
    if schema == "legacy":
        return LEGACY_KEYS
    return MERGED_KEYS


def aliases_for_key(key: str):
    return KEY_ALIASES.get(key, [key])


def parse_key_value_pairs(text: str, keys):
    parsed = {}
    for key in keys:
        alias_pattern = "|".join(re.escape(alias) for alias in aliases_for_key(key))
        pattern = rf"\[?\s*(?:{alias_pattern})\s*\]?\s*[:：]\s*([A-Za-z]+)"
        match = re.search(pattern, text, re.I)
        parsed[key] = normalize_label(match.group(1)) if match else "fail"
    return parsed


def merge_legacy_to_merged(parsed_legacy):
    merged = {"Video Quality": parsed_legacy.get("Video Quality", "fail")}
    for key, legacy_keys in MERGE_GROUPS.items():
        values = [parsed_legacy.get(legacy_key, "fail") for legacy_key in legacy_keys]
        if all(value == "yes" for value in values):
            merged[key] = "yes"
        elif any(value == "no" for value in values):
            merged[key] = "no"
        else:
            merged[key] = "fail"
    return merged


def parse_output(text: str, metric_keys):
    text = "" if text is None else str(text)
    answer_match = re.search(r"<answer>\s*(.*?)\s*(?:</answer>|$)", text, re.S | re.I)
    body = answer_match.group(1).strip() if answer_match else text.strip()

    parsed = parse_key_value_pairs(body, metric_keys)
    if metric_keys == MERGED_KEYS and any(parsed.get(key) == "fail" for key in MERGED_KEYS):
        parsed_legacy = parse_key_value_pairs(body, LEGACY_KEYS)
        merged_legacy = merge_legacy_to_merged(parsed_legacy)
        for key in MERGED_KEYS:
            if parsed.get(key) == "fail":
                parsed[key] = merged_legacy.get(key, "fail")
    return parsed


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def calculate_metrics(outputs, metric_keys):
    stats = {key: defaultdict(int) for key in metric_keys}

    for item in outputs:
        pred_dict = parse_output(item["answer"], metric_keys)
        gt_dict = parse_output(item["ground_truth"], metric_keys)

        for key in metric_keys:
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
    for key in metric_keys:
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
    summary = {f: sum(metrics[k].get(f, 0) for k in metric_keys) for f in all_fields}
    summary["accuracy"] = safe_div(summary["correct"], summary["total"])
    summary["precision"] = safe_div(summary["tp"], summary["tp"] + summary["fp"])
    summary["recall"] = safe_div(summary["tp"], summary["tp"] + summary["fn"])
    summary["f1"] = safe_div(
        2 * summary["precision"] * summary["recall"],
        summary["precision"] + summary["recall"],
    )

    return metrics, summary


def print_table(metrics, summary, metric_keys):
    header = ["Dimension", "GT(Y/N)", "TP", "TN", "FP", "FN", "Acc", "Prec", "Rec", "F1"]
    rows = []
    for k in metric_keys:
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


def extract_first_video(sample):
    for message in sample:
        for part in message.get("content", []):
            if part.get("type") == "video":
                return part.get("video")
    return None


def extract_text(sample):
    text_parts = []
    for message in sample:
        for part in message.get("content", []):
            if part.get("type") == "text":
                text_parts.append(part.get("text", ""))
    return "\n".join(text_parts)


def resolve_eval_model_type(args, inference_model_path):
    if args.model_type == "auto":
        return infer_model_type(inference_model_path)
    return args.model_type


def resolve_backend(args, model_type: str, default_backend: str) -> str:
    if args.backend != "auto":
        return args.backend
    return default_backend


def result_paths(args):
    model_name = "-".join(args.model_path.split("/")[1:])
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f"{model_name}.json")
    metrics_path = output_path.replace(".json", "_metrics.json")
    return output_path, metrics_path


def save_metrics(metrics_path, metric_keys, metrics, summary):
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metric_schema": "legacy" if metric_keys == LEGACY_KEYS else "merged",
                "per_dim": metrics,
                "overall": summary,
            },
            f,
            indent=4,
            ensure_ascii=False,
        )


def load_cached_results(output_path):
    if not os.path.exists(output_path):
        return None
    print(f"Loading cached results from {output_path}")
    with open(output_path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            print("Warning: JSON file corrupted or empty. Starting from scratch.")
            return []


def write_results(output_path, results):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)


def build_dataloader(processor, args):
    data_module = make_rl_data_module(processor=processor, data_args=args)
    return DataLoader(
        data_module["train_dataset"],
        collate_fn=data_module["data_collator"],
        batch_size=args.batch_size,
    )


def run_eval_vllm(args, inference_model_path, output_path):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=inference_model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path=args.allowed_local_media_path,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["</answer>", "<|im_end|>"],
    )

    processor = AutoProcessor.from_pretrained(
        inference_model_path,
        trust_remote_code=True,
    )
    dataloader = build_dataloader(processor, args)

    results = []
    for batch in tqdm(dataloader, desc="Inference(vLLM)"):
        user_prompts = batch["user"]
        gt_prompts = batch["gt"]

        converted_prompts = [to_vllm_chat_format(p) for p in user_prompts]
        outputs = llm.chat(converted_prompts, sampling_params=sampling_params)
        preds = [trim_repeated_response(o.outputs[0].text) for o in outputs]

        for i, pred in enumerate(preds):
            results.append(
                {
                    "video": extract_first_video(user_prompts[i]),
                    "answer": pred,
                    "ground_truth": extract_text(gt_prompts[i]),
                }
            )

        write_results(output_path, results)

    return results


def generate_hf_one(model, processor, model_type: str, args, messages):
    if model_type == "molmo2":
        messages = normalize_molmo2_messages(messages)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        **build_template_kwargs(model_type, args),
    )
    inputs = inputs.to(model.device)

    generated_ids = model.generate(
        **inputs,
        **build_generation_kwargs(processor.tokenizer, args, model_type),
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return trim_repeated_response(output_text[0])


def run_eval_hf(args, inference_model_path, model_type: str, output_path):
    model = load_model(
        inference_model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(inference_model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    if model_type == "internvl":
        configure_internvl_processor(processor, model, args)

    dataloader = build_dataloader(processor, args)

    results = []
    for batch in tqdm(dataloader, desc="Inference(HF)"):
        user_prompts = batch["user"]
        gt_prompts = batch["gt"]

        for user_prompt, gt_prompt in zip(user_prompts, gt_prompts):
            pred = generate_hf_one(model, processor, model_type, args, user_prompt)
            results.append(
                {
                    "video": extract_first_video(user_prompt),
                    "answer": pred,
                    "ground_truth": extract_text(gt_prompt),
                }
            )

        write_results(output_path, results)

    return results


@torch.inference_mode()
def run_eval(args, allowed_model_types: set[str], default_backend: str):
    metric_keys = metric_keys_from_schema(args.metric_schema)
    inference_model_path = prepare_inference_model_dir(args.model_path)
    model_type = resolve_eval_model_type(args, inference_model_path)
    if model_type not in allowed_model_types:
        allowed = ", ".join(sorted(allowed_model_types))
        raise ValueError(
            f"This evaluation entrypoint supports {allowed}, got {model_type}."
        )

    args.model_type = model_type
    backend = resolve_backend(args, model_type, default_backend)
    output_path, metrics_path = result_paths(args)

    print(f"model_type={model_type}, backend={backend}")

    results = load_cached_results(output_path)
    if results is not None and results:
        metrics, summary = calculate_metrics(results, metric_keys)
        print_table(metrics, summary, metric_keys)
        save_metrics(metrics_path, metric_keys, metrics, summary)
        print(f"Metrics saved to {metrics_path}")
        return

    if backend == "hf":
        results = run_eval_hf(args, inference_model_path, model_type, output_path)
    else:
        results = run_eval_vllm(args, inference_model_path, output_path)

    metrics, summary = calculate_metrics(results, metric_keys)
    print_table(metrics, summary, metric_keys)

    save_metrics(metrics_path, metric_keys, metrics, summary)
    print(f"Metrics saved to {metrics_path}")
