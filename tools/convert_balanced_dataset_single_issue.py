from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


TASKS = {
    "human_distortion": {
        "question": "qn1",
        "answer_key": "has_human_distortion_issue",
        "issue_name": "Human distortion issue",
        "negative_reason": "No human distortion issue annotated.",
    },
    "physics_violation": {
        "question": "qn3",
        "answer_key": "has_physics_violation_issue",
        "issue_name": "Physics violation issue",
        "negative_reason": "No physics violation issue annotated.",
    },
    "product_consistency": {
        "question": "qn4",
        "answer_key": "has_product_consistency_issue",
        "issue_name": "Product consistency issue",
        "negative_reason": "No product consistency issue annotated.",
    },
}

DEFAULT_HUMAN_VALUE = "<video>\n"


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def ensure_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return text
    if text[-1] not in ".!?。！？":
        text += "."
    return text


def normalize_yes_no(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip().lower()
    if value == "yes":
        return "Yes"
    if value == "no":
        return "No"
    return None


def parse_json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def format_second(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{number:.1f}"
    return text.rstrip("0").rstrip(".")


def format_time_ranges(ranges: Any) -> str:
    tags = []
    if not isinstance(ranges, list):
        return ""
    for item in ranges:
        if not isinstance(item, dict):
            continue
        start = format_second(item.get("start"))
        end = format_second(item.get("end"))
        if start is not None and end is not None:
            tags.append(f"<t>{start}s-{end}s</t>")
    return " ".join(tags)


def format_final_time(record: dict[str, Any], qn: str) -> str:
    start = format_second(record.get(f"[Final]{qn}_start_sec"))
    end = format_second(record.get(f"[Final]{qn}_end_sec"))
    if start is None or end is None:
        return ""
    return f"<t>{start}s-{end}s</t>"


def video_ref(record: dict[str, Any]) -> str:
    return clean_text(record.get("video_path")) or clean_text(record.get("video_url"))


def get_answer(record: dict[str, Any], task: dict[str, str]) -> str | None:
    final_eval = parse_json_object(record.get("final_eval_result"))
    answer = normalize_yes_no(final_eval.get(task["answer_key"]))
    if answer:
        return answer
    return normalize_yes_no(record.get(f"[Final]{task['question']}"))


def get_reason(record: dict[str, Any], task: dict[str, str], answer: str) -> str:
    qn = task["question"]
    evidence = clean_text(record.get(f"[Final]{qn}_1"))
    time_tag = format_final_time(record, qn)

    if not evidence or not time_tag:
        final_eval = parse_json_object(record.get("final_eval_result"))
        if final_eval.get(task["answer_key"]) is not None:
            evidence = evidence or clean_text(final_eval.get("evidence"))
            time_tag = time_tag or format_time_ranges(final_eval.get("time_ranges"))

    if answer == "No":
        evidence = evidence or task["negative_reason"]
        return ensure_sentence(evidence)

    evidence = evidence or f"There is a {task['issue_name'].lower()}."
    prefix = f"{task['issue_name']}:"
    if time_tag:
        return ensure_sentence(f"{prefix} {time_tag} {evidence}")
    return ensure_sentence(f"{prefix} {evidence}")


def build_item(
    record: dict[str, Any],
    task: dict[str, str],
    human_value: str,
    keep_metadata: bool,
) -> dict[str, Any] | None:
    video = video_ref(record)
    if not video:
        return None

    answer = get_answer(record, task)
    if answer is None:
        return None

    reason = get_reason(record, task, answer)
    item = {
        "videos": [video],
        "conversations": [
            {
                "from": "human",
                "value": human_value,
            },
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{reason}\n</think>\n"
                    f"<answer>\n{answer}\n</answer>"
                ),
            },
        ],
    }

    if keep_metadata:
        item["metadata"] = {
            "prompt_text": clean_text(record.get("prompt_text")),
            "video_url": clean_text(record.get("video_url")),
            "first_image_url": clean_text(record.get("first_image_url")),
            "image_path": clean_text(record.get("image_path")),
            "source_file": clean_text(record.get("__source_file")),
        }

    return item


def convert_task(
    input_dir: Path,
    task_name: str,
    output_path: Path,
    include_removed: bool,
    human_value: str,
    keep_metadata: bool,
) -> Counter:
    task = TASKS[task_name]
    input_files = [input_dir / task_name / "sampled.jsonl"]
    if include_removed:
        input_files.append(input_dir / task_name / "removed_scene_jumpcut.jsonl")

    output = []
    stats = Counter()

    for input_file in input_files:
        if not input_file.exists():
            continue
        with input_file.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                stats["source_rows"] += 1
                record = json.loads(line)
                item = build_item(record, task, human_value, keep_metadata)
                if item is None:
                    stats["skipped"] += 1
                    continue
                answer = item["conversations"][1]["value"].split("<answer>\n", 1)[1]
                answer = answer.split("\n</answer>", 1)[0]
                stats[answer] += 1
                output.append(item)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    stats["items"] = len(output)
    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/balanced_dataset")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--prefix", default="balanced_dataset_")
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=sorted(TASKS),
        default=sorted(TASKS),
    )
    parser.add_argument(
        "--include-removed",
        action="store_true",
        help="Also include removed_scene_jumpcut.jsonl files.",
    )
    parser.add_argument(
        "--human-value",
        default=DEFAULT_HUMAN_VALUE,
        help="Human message value. Keep <video> if videos are present.",
    )
    parser.add_argument("--keep-metadata", action="store_true")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    for task_name in args.tasks:
        output_path = output_dir / f"{args.prefix}{task_name}.json"
        stats = convert_task(
            input_dir=input_dir,
            task_name=task_name,
            output_path=output_path,
            include_removed=args.include_removed,
            human_value=args.human_value,
            keep_metadata=args.keep_metadata,
        )
        print(f"{task_name}: saved {output_path}")
        print(dict(stats))


if __name__ == "__main__":
    main()
