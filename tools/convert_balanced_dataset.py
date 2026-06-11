from __future__ import annotations

import argparse
import json
import re
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Any


DIMENSIONS = (
    "Video Quality",
    "Motion & Interaction",
    "Prompt Alignment",
)

ISSUES = OrderedDict(
    [
        (
            "human_distortion",
            {
                "question": "qn1",
                "name": "Human distortion issue",
                "dimension": "Motion & Interaction",
            },
        ),
        (
            "physics_violation",
            {
                "question": "qn3",
                "name": "Physics violation issue",
                "dimension": "Motion & Interaction",
            },
        ),
        (
            "product_consistency",
            {
                "question": "qn4",
                "name": "Product consistency issue",
                "dimension": "Prompt Alignment",
            },
        ),
        (
            "visual_text",
            {
                "question": "qn5",
                "name": "Visual text issue",
                "dimension": "Video Quality",
            },
        ),
    ]
)

FINAL_EVAL_KEYS = {
    "has_human_distortion_issue": "human_distortion",
    "has_physics_violation_issue": "physics_violation",
    "has_product_consistency_issue": "product_consistency",
}

DEFAULT_HUMAN_VALUE = "<video>\n"


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


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_evidence_key(value: str) -> str:
    normalized = value.lower().strip()
    normalized = re.sub(r"<t>.*?</t>", " ", normalized)
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def format_second(value: Any) -> str | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    text = f"{number:.1f}"
    return text.rstrip("0").rstrip(".")


def format_time_ranges(
    ranges: Any,
    fallback_start: Any = None,
    fallback_end: Any = None,
) -> str:
    tags = []
    if isinstance(ranges, list):
        for item in ranges:
            if not isinstance(item, dict):
                continue
            start = format_second(item.get("start"))
            end = format_second(item.get("end"))
            if start is not None and end is not None:
                tags.append(f"<t>{start}s-{end}s</t>")

    if not tags:
        start = format_second(fallback_start)
        end = format_second(fallback_end)
        if start is not None and end is not None:
            tags.append(f"<t>{start}s-{end}s</t>")

    return " ".join(tags)


def time_tag_score(time_tag: str) -> float:
    score = 0.0
    for start, end in re.findall(r"<t>([0-9.]+)s-([0-9.]+)s</t>", time_tag):
        try:
            score += max(0.0, float(end) - float(start))
        except ValueError:
            continue
    return score


def final_eval_for_issue(record: dict[str, Any]) -> tuple[str | None, dict[str, Any]]:
    final_eval = parse_json_object(record.get("final_eval_result"))
    for key, issue in FINAL_EVAL_KEYS.items():
        if key in final_eval:
            return issue, final_eval
    return None, {}


def record_issue_annotation(
    record: dict[str, Any],
    issue: str,
) -> tuple[str | None, str, str]:
    config = ISSUES[issue]
    qn = config["question"]

    decision = normalize_yes_no(record.get(f"[Final]{qn}"))
    evidence = clean_text(record.get(f"[Final]{qn}_1"))
    time_tag = format_time_ranges(
        [],
        record.get(f"[Final]{qn}_start_sec"),
        record.get(f"[Final]{qn}_end_sec"),
    )

    final_issue, final_eval = final_eval_for_issue(record)
    if final_issue == issue:
        for key, mapped_issue in FINAL_EVAL_KEYS.items():
            if mapped_issue != issue or key not in final_eval:
                continue
            decision = normalize_yes_no(final_eval.get(key)) or decision
            evidence = clean_text(final_eval.get("evidence")) or evidence
            time_tag = format_time_ranges(
                final_eval.get("time_ranges"),
                record.get(f"[Final]{qn}_start_sec"),
                record.get(f"[Final]{qn}_end_sec"),
            )
            break

    return decision, evidence, time_tag


def iter_source_records(input_dir: Path, include_removed: bool):
    file_names = ["sampled.jsonl"]
    if include_removed:
        file_names.append("removed_scene_jumpcut.jsonl")

    for category_dir in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        for file_name in file_names:
            path = category_dir / file_name
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as f:
                for line_number, line in enumerate(f, 1):
                    if not line.strip():
                        continue
                    yield category_dir.name, path, line_number, json.loads(line)


def video_key(record: dict[str, Any]) -> str:
    return clean_text(record.get("video_path")) or clean_text(record.get("video_url"))


def empty_aggregate(record: dict[str, Any], category: str) -> dict[str, Any]:
    key = video_key(record)
    return {
        "video": key,
        "prompt_text": clean_text(record.get("prompt_text")),
        "video_url": clean_text(record.get("video_url")),
        "first_image_url": clean_text(record.get("first_image_url")),
        "image_path": clean_text(record.get("image_path")),
        "categories": {category},
        "source_files": set(),
        "issues": {issue: OrderedDict() for issue in ISSUES},
    }


def add_record_to_aggregate(
    aggregate: dict[str, Any],
    record: dict[str, Any],
    category: str,
    source_file: Path,
):
    aggregate["categories"].add(category)
    aggregate["source_files"].add(str(source_file))

    for field in ("prompt_text", "video_url", "first_image_url", "image_path"):
        if not aggregate.get(field):
            aggregate[field] = clean_text(record.get(field))

    for issue in ISSUES:
        decision, evidence, time_tag = record_issue_annotation(record, issue)
        if decision != "Yes":
            continue

        if not evidence:
            evidence = f"There is a {ISSUES[issue]['name'].lower()}."

        prefix = f"{time_tag} " if time_tag else ""
        fragment = f"{ISSUES[issue]['name']}: {prefix}{evidence}".strip()
        key = normalize_evidence_key(evidence) or normalize_evidence_key(fragment)
        current = aggregate["issues"][issue].get(key)
        if current is None or time_tag_score(fragment) > time_tag_score(current):
            aggregate["issues"][issue][key] = fragment


def build_dimension_reasons(
    issues: dict[str, OrderedDict[str, str]],
) -> dict[str, list[str]]:
    reasons = {dimension: [] for dimension in DIMENSIONS}
    for issue, fragments_by_key in issues.items():
        dimension = ISSUES[issue]["dimension"]
        reasons[dimension].extend(fragments_by_key.values())
    return reasons


def build_gpt_value(issues: dict[str, OrderedDict[str, str]]) -> str:
    dimension_reasons = build_dimension_reasons(issues)

    answer = {}
    think_lines = []
    for dimension in DIMENSIONS:
        fragments = dimension_reasons[dimension]
        answer[dimension] = "No" if fragments else "Yes"
        reason = " ".join(fragments) if fragments else "Good."
        think_lines.append(f"{dimension}: {reason}")

    answer_line = " ".join(f"{dimension}: {answer[dimension]}." for dimension in DIMENSIONS)
    return (
        "<think>\n"
        + "\n".join(think_lines)
        + "\n</think>\n<answer>\n"
        + answer_line
        + "\n</answer>"
    )


def build_output_item(
    aggregate: dict[str, Any],
    human_value: str,
    keep_metadata: bool,
) -> dict[str, Any]:
    item = {
        "videos": [aggregate["video"]],
        "conversations": [
            {
                "from": "human",
                "value": human_value,
            },
            {
                "from": "gpt",
                "value": build_gpt_value(aggregate["issues"]),
            },
        ],
    }

    if keep_metadata:
        item["metadata"] = {
            "prompt_text": aggregate.get("prompt_text", ""),
            "video_url": aggregate.get("video_url", ""),
            "first_image_url": aggregate.get("first_image_url", ""),
            "image_path": aggregate.get("image_path", ""),
            "categories": sorted(aggregate.get("categories", [])),
            "source_files": sorted(aggregate.get("source_files", [])),
        }

    return item


def convert_rows(
    input_dir: Path,
    include_removed: bool,
    human_value: str,
    keep_metadata: bool,
) -> tuple[list[dict[str, Any]], Counter, int]:
    output = []
    stats = Counter()
    skipped = 0

    for category, source_file, _line_number, record in iter_source_records(
        input_dir,
        include_removed,
    ):
        if not video_key(record):
            skipped += 1
            continue

        aggregate = empty_aggregate(record, category)
        add_record_to_aggregate(aggregate, record, category, source_file)
        output.append(build_output_item(aggregate, human_value, keep_metadata))
        stats.update([category])

    return output, stats, skipped


def convert_unique(
    input_dir: Path,
    include_removed: bool,
    human_value: str,
    keep_metadata: bool,
) -> tuple[list[dict[str, Any]], Counter, int]:
    aggregates: OrderedDict[str, dict[str, Any]] = OrderedDict()
    stats = Counter()
    skipped = 0

    for category, source_file, _line_number, record in iter_source_records(
        input_dir,
        include_removed,
    ):
        key = video_key(record)
        if not key:
            skipped += 1
            continue

        if key not in aggregates:
            aggregates[key] = empty_aggregate(record, category)
        add_record_to_aggregate(aggregates[key], record, category, source_file)
        stats.update([category])

    output = [
        build_output_item(aggregate, human_value, keep_metadata)
        for aggregate in aggregates.values()
    ]
    return output, stats, skipped


def summarize_answers(output: list[dict[str, Any]]) -> Counter:
    counts = Counter()
    pattern = re.compile(
        r"(Video Quality|Motion & Interaction|Prompt Alignment):\s*(Yes|No)"
    )
    for item in output:
        gpt_value = item["conversations"][1]["value"]
        labels = tuple(label for _dimension, label in pattern.findall(gpt_value))
        counts.update([labels])
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="data/balanced_dataset")
    parser.add_argument("--output", default="data/balanced_dataset_merged.json")
    parser.add_argument(
        "--mode",
        choices=("unique", "rows"),
        default="unique",
        help="unique merges duplicate videos; rows keeps one output item per source row.",
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
    parser.add_argument(
        "--keep-metadata",
        action="store_true",
        help="Keep source prompt/url/category fields under metadata.",
    )
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_path = Path(args.output)

    if args.mode == "rows":
        output, source_stats, skipped = convert_rows(
            input_dir,
            args.include_removed,
            args.human_value,
            args.keep_metadata,
        )
    else:
        output, source_stats, skipped = convert_unique(
            input_dir,
            args.include_removed,
            args.human_value,
            args.keep_metadata,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        if args.indent < 0:
            json.dump(output, f, ensure_ascii=False)
        else:
            json.dump(output, f, ensure_ascii=False, indent=args.indent)

    print(f"saved: {output_path}")
    print(f"items: {len(output)}")
    print(f"skipped_no_video: {skipped}")
    print(f"source_rows: {dict(source_stats)}")
    print(f"answer_patterns: {dict(summarize_answers(output))}")


if __name__ == "__main__":
    main()
