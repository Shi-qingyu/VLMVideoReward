import argparse
import json
import re
from pathlib import Path
from typing import Any


TIME_TAG_PATTERN = re.compile(r"<t>.*?</t>", re.DOTALL)


def strip_time_tags(text: str) -> tuple[str, int]:
    cleaned, count = TIME_TAG_PATTERN.subn(" ", text)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r" *\n *", "\n", cleaned)
    return cleaned.strip(), count


def clean_item(item: Any) -> int:
    if not isinstance(item, dict):
        return 0

    changed = 0
    conversations = item.get("conversations")
    if not isinstance(conversations, list):
        return 0

    for message in conversations:
        if not isinstance(message, dict):
            continue
        if message.get("from") != "gpt":
            continue
        value = message.get("value")
        if not isinstance(value, str):
            continue
        cleaned, count = strip_time_tags(value)
        if count:
            message["value"] = cleaned
            changed += count
    return changed


def load_json_or_jsonl(path: Path) -> tuple[Any, bool]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()], True

    with path.open("r", encoding="utf-8") as f:
        return json.load(f), False


def save_json_or_jsonl(data: Any, path: Path, is_jsonl: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        if is_jsonl:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        else:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove <t>...</t> spans from gpt conversation values."
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "-o",
        "--output-file",
        type=Path,
        help="Output path. If omitted with --in-place, overwrites input_file.",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite input_file instead of writing a separate output file.",
    )
    args = parser.parse_args()

    if args.in_place and args.output_file is not None:
        parser.error("--in-place and --output-file cannot be used together")
    if not args.in_place and args.output_file is None:
        parser.error("provide --output-file or use --in-place")

    data, is_jsonl = load_json_or_jsonl(args.input_file)
    items = data if isinstance(data, list) else [data]
    removed = sum(clean_item(item) for item in items)

    output_file = args.input_file if args.in_place else args.output_file
    save_json_or_jsonl(data, output_file, is_jsonl)
    print(f"Removed {removed} time-tag span(s). Saved to: {output_file}")


if __name__ == "__main__":
    main()
