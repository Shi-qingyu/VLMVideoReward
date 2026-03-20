import os
import re
import json
import argparse
from collections import defaultdict
from tqdm import tqdm

from qwenvl.dataset.eval_data import load_eval_data
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
args = parser.parse_args()


EXPECTED_KEYS = [
    "Video Quality",
    "Subject Movement",
    "Physical Interaction",
    "Cause-Effect",
    "Subject Existence",
    "Object Existence",
    "Subject-Object Interaction",
]


def normalize_label(text: str) -> str:
    if text is None:
        return "fail"

    text = str(text).strip().lower()

    text = text.replace("good", "yes").replace("bad", "no")

    if text in ["yes", "y", "true", "1"]:
        return "yes"
    if text in ["no", "n", "false", "0"]:
        return "no"
    return "fail"


def safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def parse_output(text: str):
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.S)
    think_content = think_match.group(1).strip() if think_match else ""

    answer_match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S)
    answer_text = answer_match.group(1).strip() if answer_match else ""

    pairs = re.findall(r"([\w\- ]+):\s*([^.\n]+)\.?", answer_text)
    raw_dict = {k.strip(): v.strip() for k, v in pairs}
    answer_dict = {k: normalize_label(raw_dict.get(k, "Fail")) for k in EXPECTED_KEYS}

    return think_content, answer_dict


def calculate_score(outputs):
    score = 0
    cnt = 0

    for output in outputs:
        answer = output["answer"]
        gt = output["ground_truth"]

        _, pred_dict = parse_output(answer)
        _, gt_dict = parse_output(gt)

        for key in EXPECTED_KEYS:
            if gt_dict[key].lower() == "fail":
                # print(gt_dict[key])
                continue
            
            if pred_dict[key] == gt_dict[key]:
                score += 1
            cnt += 1

    return safe_div(score, cnt)


def calculate_detailed_metrics(outputs):
    """
    - accuracy
    - precision
    - recall
    - gt_yes / gt_no
    - tp / tn / fp / fn
    """
    stats = {
        key: {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "gt_yes": 0,
            "gt_no": 0,
            "pred_yes": 0,
            "pred_no": 0,
            "correct": 0,
            "total": 0,
        }
        for key in EXPECTED_KEYS
    }

    for output in outputs:
        answer = output["answer"]
        gt = output["ground_truth"]

        _, pred_dict = parse_output(answer)
        _, gt_dict = parse_output(gt)

        for key in EXPECTED_KEYS:
            pred = pred_dict[key]
            gt_label = gt_dict[key]

            if gt_label not in ["yes", "no"]:
                continue

            stats[key]["total"] += 1

            if gt_label == "yes":
                stats[key]["gt_yes"] += 1
            elif gt_label == "no":
                stats[key]["gt_no"] += 1

            if pred == "yes":
                stats[key]["pred_yes"] += 1
            elif pred == "no":
                stats[key]["pred_no"] += 1

            if pred == gt_label:
                stats[key]["correct"] += 1

            if gt_label == "yes" and pred == "yes":
                stats[key]["tp"] += 1
            elif gt_label == "no" and pred == "no":
                stats[key]["tn"] += 1
            elif gt_label == "no" and pred == "yes":
                stats[key]["fp"] += 1
            elif gt_label == "yes" and pred == "no":
                stats[key]["fn"] += 1
            else:
                if gt_label == "yes":
                    stats[key]["fn"] += 1

    metrics = {}
    for key, s in stats.items():
        accuracy = safe_div(s["correct"], s["total"])
        precision = safe_div(s["tp"], s["tp"] + s["fp"])
        recall = safe_div(s["tp"], s["tp"] + s["fn"])
        f1 = safe_div(2 * precision * recall, precision + recall)

        metrics[key] = {
            **s,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def build_summary(metrics):
    total_tp = sum(v["tp"] for v in metrics.values())
    total_tn = sum(v["tn"] for v in metrics.values())
    total_fp = sum(v["fp"] for v in metrics.values())
    total_fn = sum(v["fn"] for v in metrics.values())
    total_correct = sum(v["correct"] for v in metrics.values())
    total_count = sum(v["total"] for v in metrics.values())
    total_gt_yes = sum(v["gt_yes"] for v in metrics.values())
    total_gt_no = sum(v["gt_no"] for v in metrics.values())

    accuracy = safe_div(total_correct, total_count)
    precision = safe_div(total_tp, total_tp + total_fp)
    recall = safe_div(total_tp, total_tp + total_fn)
    f1 = safe_div(2 * precision * recall, precision + recall)

    return {
        "tp": total_tp,
        "tn": total_tn,
        "fp": total_fp,
        "fn": total_fn,
        "gt_yes": total_gt_yes,
        "gt_no": total_gt_no,
        "correct": total_correct,
        "total": total_count,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def format_percent(x: float) -> str:
    return f"{x * 100:6.2f}%"


def print_metrics_table(metrics, summary):
    headers = [
        "Dimension",
        "GT Yes",
        "GT No",
        "TP",
        "TN",
        "FP",
        "FN",
        "Acc",
        "Prec",
        "Recall",
        "F1",
    ]

    rows = []
    for key in EXPECTED_KEYS:
        m = metrics[key]
        rows.append([
            key,
            str(m["gt_yes"]),
            str(m["gt_no"]),
            str(m["tp"]),
            str(m["tn"]),
            str(m["fp"]),
            str(m["fn"]),
            format_percent(m["accuracy"]),
            format_percent(m["precision"]),
            format_percent(m["recall"]),
            format_percent(m["f1"]),
        ])

    rows.append([
        "OVERALL(micro)",
        str(summary["gt_yes"]),
        str(summary["gt_no"]),
        str(summary["tp"]),
        str(summary["tn"]),
        str(summary["fp"]),
        str(summary["fn"]),
        format_percent(summary["accuracy"]),
        format_percent(summary["precision"]),
        format_percent(summary["recall"]),
        format_percent(summary["f1"]),
    ])

    col_widths = []
    for i, h in enumerate(headers):
        max_len = len(h)
        for row in rows:
            max_len = max(max_len, len(row[i]))
        col_widths.append(max_len)

    def make_sep(char="-"):
        return "+" + "+".join(char * (w + 2) for w in col_widths) + "+"

    def make_row(items):
        return "| " + " | ".join(items[i].ljust(col_widths[i]) for i in range(len(items))) + " |"

    print("\n" + "=" * 120)
    print("Evaluation Metrics")
    print("=" * 120)
    print(make_sep("="))
    print(make_row(headers))
    print(make_sep("="))

    for i, row in enumerate(rows):
        print(make_row(row))
        if i == len(rows) - 2:
            print(make_sep("="))
        else:
            print(make_sep("-"))

    print()


def save_metrics(metrics, summary, output_path):
    metric_path = output_path.replace(".json", "_metrics.json")
    dump_obj = {
        "per_dimension": metrics,
        "overall_micro": summary,
    }
    with open(metric_path, "w", encoding="utf-8") as f:
        json.dump(dump_obj, f, indent=4, ensure_ascii=False)

    print(f"Detailed metrics saved to: {metric_path}")


def run_inference(args):
    model_name = args.model_path.split("/")[1]
    output_dir = "eval_results"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, model_name + ".json")

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            outputs = json.load(f)

        overall_score = calculate_score(outputs)
        metrics = calculate_detailed_metrics(outputs)
        summary = build_summary(metrics)

        print(f"\nCached result found: {output_path}")
        print(f"Overall Accuracy: {overall_score:.6f}")
        print_metrics_table(metrics, summary)
        save_metrics(metrics, summary, output_path)
        return

    # default: Load the model on the available device(s)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_path, dtype="auto", device_map="auto"
    )

    try:
        processor = AutoProcessor.from_pretrained(args.model_path)
    except Exception:
        processor = AutoProcessor.from_pretrained(os.path.dirname(args.model_path))

    dataset, ground_truths = load_eval_data("./data/eval.json")

    outputs = []
    for idx, (data, gt) in enumerate(
        tqdm(
            zip(dataset, ground_truths),
            total=len(dataset),
            desc="Evaluating",
            ncols=100
        ),
        start=1
    ):
        inputs = [data]

        # Preparation for inference
        inputs = processor.apply_chat_template(
            inputs,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)

        # Inference
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        outputs.append(
            {
                "video_path": data["content"][0]["video"],
                "answer": output_text[0],
                "ground_truth": gt,
            }
        )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4, ensure_ascii=False)

    overall_score = calculate_score(outputs)
    metrics = calculate_detailed_metrics(outputs)
    summary = build_summary(metrics)

    print(f"\nResults saved to: {output_path}")
    print(f"Overall Accuracy: {overall_score:.6f}")
    print_metrics_table(metrics, summary)
    save_metrics(metrics, summary, output_path)


def main(args):
    run_inference(args)


if __name__ == "__main__":
    main(args)