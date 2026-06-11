from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


DEFAULT_INPUTS = [
    "data/human_distortion_sampled.json",
    "data/physics_violation_sampled.json",
    "data/product_consistency_sampled.json",
]


def answer_label(item: dict[str, Any]) -> str:
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return "UNKNOWN"
    text = str(conversations[-1].get("value", ""))
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I)
    return re.sub(r"\s+", " ", match.group(1)).strip() if match else "UNKNOWN"


def split_counts(group_sizes: dict[str, int], test_size: int) -> dict[str, int]:
    total = sum(group_sizes.values())
    if test_size > total:
        raise ValueError(f"test_size={test_size} exceeds total items={total}")

    raw = {
        label: (count * test_size / total)
        for label, count in group_sizes.items()
    }
    counts = {
        label: min(int(value), group_sizes[label])
        for label, value in raw.items()
    }

    remaining = test_size - sum(counts.values())
    labels_by_remainder = sorted(
        group_sizes,
        key=lambda label: (raw[label] - int(raw[label]), group_sizes[label]),
        reverse=True,
    )
    while remaining > 0:
        progressed = False
        for label in labels_by_remainder:
            if counts[label] >= group_sizes[label]:
                continue
            counts[label] += 1
            remaining -= 1
            progressed = True
            if remaining == 0:
                break
        if not progressed:
            raise ValueError("Could not allocate requested test size.")

    return counts


def output_paths(input_path: Path) -> tuple[Path, Path]:
    stem = input_path.stem
    return (
        input_path.with_name(f"{stem}_train{input_path.suffix}"),
        input_path.with_name(f"{stem}_test{input_path.suffix}"),
    )


def split_dataset(
    input_path: Path,
    test_size: int,
    seed: int,
    indent: int,
) -> dict[str, Any]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{input_path} must contain a JSON list.")

    rng = random.Random(seed)
    indexed_groups: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(data):
        indexed_groups[answer_label(item)].append(index)

    for indices in indexed_groups.values():
        rng.shuffle(indices)

    per_label_test = split_counts(
        {label: len(indices) for label, indices in indexed_groups.items()},
        test_size,
    )

    test_indices = set()
    for label, count in per_label_test.items():
        test_indices.update(indexed_groups[label][:count])

    train = [item for index, item in enumerate(data) if index not in test_indices]
    test = [item for index, item in enumerate(data) if index in test_indices]
    rng.shuffle(train)
    rng.shuffle(test)

    train_path, test_path = output_paths(input_path)
    with train_path.open("w", encoding="utf-8") as f:
        json.dump(train, f, ensure_ascii=False, indent=indent)
    with test_path.open("w", encoding="utf-8") as f:
        json.dump(test, f, ensure_ascii=False, indent=indent)

    return {
        "input": str(input_path),
        "train": str(train_path),
        "test": str(test_path),
        "source_items": len(data),
        "train_items": len(train),
        "test_items": len(test),
        "source_labels": dict(Counter(answer_label(item) for item in data)),
        "train_labels": dict(Counter(answer_label(item) for item in train)),
        "test_labels": dict(Counter(answer_label(item) for item in test)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--test-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indent", type=int, default=2)
    args = parser.parse_args()

    for path in args.inputs:
        summary = split_dataset(
            input_path=Path(path),
            test_size=args.test_size,
            seed=args.seed,
            indent=args.indent,
        )
        print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
