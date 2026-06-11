from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_TASK = "physics_violation"
TASKS = {
    "human_distortion": {
        "answer_key": "has_human_distortion_issue",
        "question_key": "qn1",
        "input": "balanced_dataset/human_distortion/sampled.jsonl",
        "output": "data/human_distortion_sampled.json",
        "issue_name": "Human distortion issue",
        "negative_reason": "No human distortion issue is annotated.",
        "question": (
            "<video>\n"
            "You are evaluating an AI-generated video. Does this video contain any "
            "human distortion or anatomically implausible human appearance/motion, "
            "such as deformed hands or fingers, twisted limbs, extra or missing body "
            "parts, unnatural faces, or impossible human poses? Provide brief evidence "
            'in <think> and answer only "Yes" or "No" in <answer>.'
        ),
    },
    "physics_violation": {
        "answer_key": "has_physics_violation_issue",
        "question_key": "qn3",
        "input": "balanced_dataset/physics_violation/sampled.jsonl",
        "output": "data/physics_violation_sampled.json",
        "issue_name": "Physics violation issue",
        "negative_reason": "No physics violation issue is annotated.",
        "question": (
            "<video>\n"
            "You are evaluating an AI-generated video. Does this video contain any "
            "violation of physical laws or physically implausible motion/interactions, "
            "such as floating objects, impossible forces, discontinuous motion, "
            "unnatural deformation, or inconsistent object dynamics? Provide brief "
            'evidence in <think> and answer only "Yes" or "No" in <answer>.'
        ),
    },
    "product_consistency": {
        "answer_key": "has_product_consistency_issue",
        "question_key": "qn4",
        "input": "balanced_dataset/product_consistency/sampled.jsonl",
        "output": "data/product_consistency_sampled.json",
        "issue_name": "Product consistency issue",
        "negative_reason": "No product consistency issue is annotated.",
        "question": (
            "<video>\n"
            "You are evaluating an AI-generated product video. Does this video contain "
            "any product consistency issue, such as the product's shape, color, texture, "
            "logo, structure, or key details changing unnaturally across frames or "
            "differing from the intended/reference product? Provide brief evidence "
            'in <think> and answer only "Yes" or "No" in <answer>.'
        ),
    },
}


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
    normalized = str(value).strip().lower()
    if normalized == "yes":
        return "Yes"
    if normalized == "no":
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


def final_eval(record: dict[str, Any]) -> dict[str, Any]:
    for key in ("final_eval_result", "final_eval_result_raw"):
        parsed = parse_json_object(record.get(key))
        if parsed:
            return parsed
    return {}


def answer_payload(record: dict[str, Any]) -> dict[str, Any]:
    payload = parse_json_object(record.get("Answer"))
    for key in ("dataMap", "data"):
        nested = payload.get(key)
        if isinstance(nested, dict):
            return nested
    return {}


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


def format_time_range_from_fields(source: dict[str, Any], qn: str) -> str:
    start = format_second(source.get(f"[Final]{qn}_start_sec", source.get(f"{qn}_start_sec")))
    end = format_second(source.get(f"[Final]{qn}_end_sec", source.get(f"{qn}_end_sec")))
    if start is None or end is None:
        return ""
    return f"<t>{start}s-{end}s</t>"


def video_ref(record: dict[str, Any], video_field: str) -> str:
    video_path = clean_text(record.get("video_path"))
    video_url = clean_text(record.get("video_url"))
    if video_field == "path":
        return video_path
    if video_field == "url":
        return video_url
    return video_path or video_url


def get_decision(record: dict[str, Any], task: dict[str, str]) -> str | None:
    eval_result = final_eval(record)
    qn = task["question_key"]

    decision = normalize_yes_no(eval_result.get(task["answer_key"]))
    if decision:
        return decision

    decision = normalize_yes_no(record.get(f"[Final]{qn}"))
    if decision:
        return decision

    return normalize_yes_no(answer_payload(record).get(qn))


def get_evidence(record: dict[str, Any], task: dict[str, str]) -> str:
    eval_result = final_eval(record)
    qn = task["question_key"]

    evidence = clean_text(eval_result.get("evidence"))
    if evidence:
        return evidence

    evidence = clean_text(record.get(f"[Final]{qn}_1"))
    if evidence:
        return evidence

    return clean_text(answer_payload(record).get(f"{qn}_1"))


def get_time_tag(record: dict[str, Any], task: dict[str, str]) -> str:
    eval_result = final_eval(record)
    qn = task["question_key"]

    time_tag = format_time_ranges(eval_result.get("time_ranges"))
    if time_tag:
        return time_tag

    time_tag = format_time_range_from_fields(record, qn)
    if time_tag:
        return time_tag

    return format_time_range_from_fields(answer_payload(record), qn)


def build_reason(record: dict[str, Any], decision: str, task: dict[str, str]) -> str:
    evidence = get_evidence(record, task)
    time_tag = get_time_tag(record, task)

    if decision == "No":
        return ensure_sentence(evidence or task["negative_reason"])

    evidence = evidence or f"There is a {task['issue_name'].lower()}."
    if time_tag:
        return ensure_sentence(f"{task['issue_name']}: {time_tag} {evidence}")
    return ensure_sentence(f"{task['issue_name']}: {evidence}")


def build_item(
    record: dict[str, Any],
    question: str,
    task: dict[str, str],
    video_field: str,
    keep_metadata: bool,
    line_number: int,
) -> dict[str, Any] | None:
    video = video_ref(record, video_field)
    decision = get_decision(record, task)
    if not video or decision is None:
        return None

    item = {
        "videos": [video],
        "conversations": [
            {
                "from": "human",
                "value": question,
            },
            {
                "from": "gpt",
                "value": (
                    f"<think>\n{build_reason(record, decision, task)}\n</think>\n"
                    f"<answer>\n{decision}\n</answer>"
                ),
            },
        ],
    }

    if keep_metadata:
        item["metadata"] = {
            "source_line": line_number,
            "source_file": clean_text(record.get("__source_file")),
            "prompt_text": clean_text(record.get("prompt_text")),
            "video_path": clean_text(record.get("video_path")),
            "video_url": clean_text(record.get("video_url")),
            "first_image_url": clean_text(record.get("first_image_url")),
            "image_path": clean_text(record.get("image_path")),
            "video_duration_sec": record.get("video_duration_sec"),
        }

    return item


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if line.strip():
                yield line_number, json.loads(line)


def convert(
    input_path: Path,
    question: str,
    task: dict[str, str],
    video_field: str,
    keep_metadata: bool,
) -> tuple[list[dict[str, Any]], Counter]:
    output = []
    stats = Counter()

    for line_number, record in iter_jsonl(input_path):
        stats["source_rows"] += 1
        item = build_item(record, question, task, video_field, keep_metadata, line_number)
        if item is None:
            if not video_ref(record, video_field):
                stats["skipped_no_video"] += 1
            if get_decision(record, task) is None:
                stats["skipped_no_answer"] += 1
            continue

        answer = item["conversations"][1]["value"].split("<answer>\n", 1)[1]
        answer = answer.split("\n</answer>", 1)[0]
        stats[answer] += 1
        output.append(item)

    stats["items"] = len(output)
    return output, stats


def write_output(output: list[dict[str, Any]], output_path: Path, jsonl: bool, indent: int):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if jsonl:
            for item in output:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        elif indent < 0:
            json.dump(output, f, ensure_ascii=False)
        else:
            json.dump(output, f, ensure_ascii=False, indent=indent)


def main():
    parser = argparse.ArgumentParser(
        description="Convert a balanced_dataset single-issue sampled.jsonl into this repo's conversation JSON format.",
    )
    parser.add_argument("--task", choices=sorted(TASKS), default=DEFAULT_TASK)
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--question")
    parser.add_argument(
        "--video-field",
        choices=("auto", "path", "url"),
        default="auto",
        help="auto uses video_path first, then video_url.",
    )
    parser.add_argument("--keep-metadata", action="store_true")
    parser.add_argument("--jsonl", action="store_true", help="Write JSONL instead of a JSON list.")
    parser.add_argument("--indent", type=int, default=2, help="Use a negative value for compact JSON.")
    args = parser.parse_args()

    task = TASKS[args.task]
    input_path = Path(args.input or task["input"])
    output_path = Path(args.output or task["output"])
    question = args.question or task["question"]

    output, stats = convert(
        input_path=input_path,
        question=question,
        task=task,
        video_field=args.video_field,
        keep_metadata=args.keep_metadata,
    )
    write_output(output, output_path, args.jsonl, args.indent)

    print(f"saved: {output_path}")
    print(dict(stats))


if __name__ == "__main__":
    main()
