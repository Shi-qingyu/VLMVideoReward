#!/usr/bin/env python3
"""Annotate VideoReward JSON samples with Qwen-VL video captions.

The script inserts a generated caption at the beginning of the assistant
``<think>`` block, then preserves the original reasoning content.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CAPTION_PROMPT = "Describe this video in one concise English sentence."

CAPTION_RE = re.compile(r"^\s*Video Caption\s*:", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Qwen3-VL generated video captions to assistant <think> blocks.",
    )
    parser.add_argument(
        "--input",
        default="data/train_t_merged_unique.json",
        help="Input VideoReward JSON file.",
    )
    parser.add_argument(
        "--output",
        default="data/train_t_merged_unique_qwen3vl8b_captioned.json",
        help="Output JSON file to write.",
    )
    parser.add_argument(
        "--model-path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF model id or local Qwen-VL checkpoint path.",
    )
    parser.add_argument(
        "--model-type",
        default="qwen3vl",
        choices=["auto", "qwen3vl", "qwen2.5vl", "qwen2vl"],
    )
    parser.add_argument(
        "--backend",
        default="vllm",
        choices=["hf", "vllm"],
        help="Generation backend. vLLM is faster for large batches when available.",
    )
    parser.add_argument(
        "--video-root",
        action="append",
        default=None,
        help=(
            "Root joined with relative video paths. Can be repeated. "
            "Defaults to ./data."
        ),
    )
    parser.add_argument("--caption-prompt", default=DEFAULT_CAPTION_PROMPT)
    parser.add_argument("--caption-prefix", default="Video Caption: ")
    parser.add_argument("--max-new-tokens", type=int, default=48)
    parser.add_argument("--model-max-length", type=int, default=8192)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--attn-implementation", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=int(os.environ.get("NUM_GPUS", "8")),
        help=(
            "Launch this many single-GPU shard workers. Use --num-gpus 1 to run "
            "inside the current process."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        default=os.environ.get("GPU_IDS"),
        help="Comma-separated GPU ids for shard workers. Defaults to 0..num_gpus-1.",
    )
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--allowed-local-media-path", default=None)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--video-max-frames", type=int, default=None)
    parser.add_argument("--video-min-pixels", type=int, default=None)
    parser.add_argument("--video-max-pixels", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-every", type=int, default=20)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file and skip already captioned samples.",
    )
    parser.add_argument(
        "--overwrite-existing-captions",
        action="store_true",
        help="Regenerate captions even if the think block already starts with Video Caption.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite output if it already exists and --resume is not set.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only validate JSON structure and video paths; do not load the model.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Skip samples with missing videos instead of failing.",
    )
    return parser.parse_args()


def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")
    return data


def write_json(path: Path, data: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    tmp_path.replace(path)


def iter_video_roots(args: argparse.Namespace) -> list[Path]:
    roots = args.video_root or ["data"]
    return [Path(root).expanduser().resolve() for root in roots]


def normalize_media_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def video_candidates(video: str, roots: Iterable[Path]) -> list[Path]:
    raw_path = Path(video).expanduser()
    if raw_path.is_absolute():
        return [raw_path]

    candidates: list[Path] = []
    for root in roots:
        candidates.append(root / raw_path)
        if raw_path.parts and raw_path.parts[0] == "videos" and root.name == "videos":
            candidates.append(root / Path(*raw_path.parts[1:]))

    candidates.append((Path.cwd() / raw_path).resolve())
    if raw_path.parts and raw_path.parts[0] != "data":
        candidates.append((Path.cwd() / "data" / raw_path).resolve())

    deduped: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key not in seen:
            deduped.append(resolved)
            seen.add(key)
    return deduped


def resolve_video_path(sample: dict[str, Any], roots: list[Path]) -> Path:
    videos = normalize_media_list(sample.get("videos"))
    if not videos:
        raise ValueError("sample has no videos field")

    candidates = video_candidates(videos[0], roots)
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    tried = ", ".join(str(candidate) for candidate in candidates[:4])
    if len(candidates) > 4:
        tried += ", ..."
    raise FileNotFoundError(f"missing video {videos[0]!r}; tried {tried}")


def assistant_message(sample: dict[str, Any]) -> dict[str, Any]:
    conversations = sample.get("conversations")
    if not isinstance(conversations, list):
        raise ValueError("sample has no conversations list")

    for message in conversations:
        if isinstance(message, dict) and message.get("from") == "gpt":
            return message
    raise ValueError("sample has no assistant/gpt message")


def has_caption(value: str) -> bool:
    think_start = value.find("<think>")
    if think_start < 0:
        return False
    rest = value[think_start + len("<think>") :]
    rest = rest.lstrip()
    return CAPTION_RE.match(rest) is not None


def clean_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"</?think>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"</?answer>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"^assistant\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^(video\s*)?caption\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.strip(" \t\r\n\"'")
    if not text:
        raise ValueError("empty caption generated")
    return text


def insert_caption(value: str, caption: str, prefix: str) -> str:
    caption_line = f"{prefix}{caption}"
    think_start = value.find("<think>")
    if think_start < 0:
        return f"<think>\n{caption_line}\n{value.lstrip()}"

    insert_at = think_start + len("<think>")
    rest = value[insert_at:]
    if rest.startswith("\r\n"):
        insert_at += 2
    elif rest.startswith("\n"):
        insert_at += 1

    return value[:insert_at] + caption_line + "\n" + value[insert_at:]


def selected_bounds(total: int, start_index: int, limit: int | None) -> tuple[int, int]:
    start = max(start_index, 0)
    stop = total if limit is None else min(total, start + max(limit, 0))
    return start, stop


def validate_shard_args(args: argparse.Namespace) -> None:
    if args.num_shards < 1:
        raise ValueError("--num-shards must be >= 1")
    if args.shard_id < 0 or args.shard_id >= args.num_shards:
        raise ValueError("--shard-id must be in [0, num_shards)")


def iter_pending_samples(
    data: list[dict[str, Any]],
    args: argparse.Namespace,
    roots: list[Path],
) -> Iterable[tuple[int, Path]]:
    validate_shard_args(args)
    start, stop = selected_bounds(len(data), args.start_index, args.limit)
    missing_count = 0

    for index in range(start, stop):
        if args.num_shards > 1 and index % args.num_shards != args.shard_id:
            continue
        sample = data[index]
        try:
            message = assistant_message(sample)
            value = str(message.get("value", ""))
            if has_caption(value) and not args.overwrite_existing_captions:
                continue
            yield index, resolve_video_path(sample, roots)
        except Exception as exc:
            if not (args.skip_missing or args.check_only):
                raise
            missing_count += 1
            if missing_count <= 10:
                print(f"[missing] index={index}: {exc}", file=sys.stderr)
            elif missing_count == 11:
                print("[missing] further missing/invalid samples omitted", file=sys.stderr)

    if missing_count:
        print(f"skipped missing/invalid samples: {missing_count}", file=sys.stderr)


def build_video_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if args.video_fps is not None:
        kwargs["fps"] = float(args.video_fps)
    if args.video_max_frames is not None:
        kwargs["nframes"] = int(args.video_max_frames)
    if args.video_min_pixels is not None:
        kwargs["min_pixels"] = int(args.video_min_pixels)
    if args.video_max_pixels is not None:
        kwargs["max_pixels"] = int(args.video_max_pixels)
    return kwargs


def build_hf_message(
    video_path: Path,
    prompt: str,
    video_kwargs: dict[str, Any],
) -> list[dict[str, Any]]:
    video_content = {"type": "video", "video": str(video_path)}
    video_content.update(video_kwargs)
    return [
        {
            "role": "user",
            "content": [
                video_content,
                {"type": "text", "text": prompt},
            ],
        }
    ]


def build_vllm_message(video_path: Path, prompt: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "video_url",
                    "video_url": {"url": f"file://{video_path}"},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]


def load_hf_model(args: argparse.Namespace):
    from inference_common import (
        infer_model_type,
        load_model,
        load_processor,
        prepare_processor,
    )

    inference_model_path = args.model_path
    model_type = (
        infer_model_type(inference_model_path)
        if args.model_type == "auto"
        else args.model_type
    )
    model = load_model(
        inference_model_path,
        model_type,
        dtype=args.dtype,
        attn_implementation=args.attn_implementation,
        device_map=args.device_map,
    )
    processor = load_processor(inference_model_path, model_type)
    prepare_processor(processor, model, model_type, args.model_max_length)
    return model, processor, model_type


def make_generation_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )


def generate_hf_captions(
    items: Iterable[tuple[int, Path]],
    args: argparse.Namespace,
) -> Iterable[tuple[int, str]]:
    import torch
    from inference_common import build_generation_kwargs

    model, processor, model_type = load_hf_model(args)
    video_kwargs = build_video_kwargs(args)
    generation_args = make_generation_args(args)

    for index, video_path in progress(items, desc="Captioning(HF)"):
        messages = build_hf_message(video_path, args.caption_prompt, video_kwargs)
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                **build_generation_kwargs(processor.tokenizer, generation_args, model_type),
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        yield index, clean_caption(output_text)


def generate_vllm_captions(
    items: Iterable[tuple[int, Path]],
    args: argparse.Namespace,
) -> Iterable[tuple[int, str]]:
    from vllm import LLM, SamplingParams

    model_path = args.model_path
    allowed_media_path = args.allowed_local_media_path
    if allowed_media_path is None:
        roots = iter_video_roots(args)
        allowed_media_path = os.path.commonpath([str(root) for root in roots])

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_seqs=args.max_num_seqs,
        limit_mm_per_prompt={"video": 1},
        allowed_local_media_path=allowed_media_path,
    )
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_new_tokens,
        stop=["<|im_end|>", "</answer>", "</think>"],
    )

    for batch in batched(progress(items, desc="Captioning(vLLM)"), args.batch_size):
        prompts = [
            build_vllm_message(video_path, args.caption_prompt)
            for _, video_path in batch
        ]
        outputs = llm.chat(prompts, sampling_params=sampling_params)
        for (index, _), output in zip(batch, outputs):
            yield index, clean_caption(output.outputs[0].text)


def batched(items: Iterable[Any], batch_size: int) -> Iterable[list[Any]]:
    batch: list[Any] = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def progress(items: Iterable[Any], desc: str = "", total: int | None = None) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, total=total)


def prepend_item(first: Any, rest: Iterable[Any]) -> Iterable[Any]:
    yield first
    yield from rest


def parse_gpu_ids(args: argparse.Namespace) -> list[str]:
    if args.gpu_ids:
        gpu_ids = [item.strip() for item in args.gpu_ids.split(",") if item.strip()]
    else:
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if visible:
            gpu_ids = [item.strip() for item in visible.split(",") if item.strip()]
        else:
            gpu_ids = [str(index) for index in range(args.num_gpus)]

    if len(gpu_ids) < args.num_gpus:
        raise ValueError(
            f"Need {args.num_gpus} GPU ids, got {len(gpu_ids)}: {','.join(gpu_ids)}"
        )
    return gpu_ids[: args.num_gpus]


def shard_output_path(output_path: Path, shard_id: int, num_shards: int) -> Path:
    return output_path.with_name(
        f"{output_path.stem}.shard{shard_id:02d}-of-{num_shards:02d}{output_path.suffix}"
    )


def add_optional_arg(cmd: list[str], name: str, value: Any) -> None:
    if value is not None:
        cmd.extend([name, str(value)])


def build_worker_command(
    args: argparse.Namespace,
    shard_id: int,
    shard_path: Path,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--input",
        args.input,
        "--output",
        str(shard_path),
        "--model-path",
        args.model_path,
        "--model-type",
        args.model_type,
        "--backend",
        args.backend,
        "--caption-prompt",
        args.caption_prompt,
        "--caption-prefix",
        args.caption_prefix,
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--model-max-length",
        str(args.model_max_length),
        "--dtype",
        args.dtype,
        "--device-map",
        args.device_map,
        "--temperature",
        str(args.temperature),
        "--top-p",
        str(args.top_p),
        "--repetition-penalty",
        str(args.repetition_penalty),
        "--no-repeat-ngram-size",
        str(args.no_repeat_ngram_size),
        "--batch-size",
        str(args.batch_size),
        "--tensor-parallel-size",
        "1",
        "--num-gpus",
        "1",
        "--num-shards",
        str(args.num_gpus),
        "--shard-id",
        str(shard_id),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--start-index",
        str(args.start_index),
        "--write-every",
        str(args.write_every),
    ]

    for root in args.video_root or []:
        cmd.extend(["--video-root", root])

    add_optional_arg(cmd, "--attn-implementation", args.attn_implementation)
    add_optional_arg(cmd, "--allowed-local-media-path", args.allowed_local_media_path)
    add_optional_arg(cmd, "--video-fps", args.video_fps)
    add_optional_arg(cmd, "--video-max-frames", args.video_max_frames)
    add_optional_arg(cmd, "--video-min-pixels", args.video_min_pixels)
    add_optional_arg(cmd, "--video-max-pixels", args.video_max_pixels)
    add_optional_arg(cmd, "--limit", args.limit)

    if args.overwrite_existing_captions:
        cmd.append("--overwrite-existing-captions")
    if args.skip_missing:
        cmd.append("--skip-missing")
    if args.force:
        cmd.append("--force")
    else:
        cmd.append("--resume")
    return cmd


def merge_shard_outputs(
    input_path: Path,
    output_path: Path,
    shard_paths: list[Path],
    args: argparse.Namespace,
) -> int:
    if args.resume and output_path.exists():
        merged = load_json(output_path)
    else:
        merged = load_json(input_path)

    merged_count = 0
    for shard_path in shard_paths:
        shard_data = load_json(shard_path)
        if len(shard_data) != len(merged):
            raise ValueError(
                f"{shard_path} length {len(shard_data)} does not match {len(merged)}"
            )

        for index, sample in enumerate(shard_data):
            source_message = assistant_message(sample)
            source_value = str(source_message.get("value", ""))
            if not has_caption(source_value):
                continue

            target_message = assistant_message(merged[index])
            target_value = str(target_message.get("value", ""))
            if has_caption(target_value) and not args.overwrite_existing_captions:
                continue
            target_message["value"] = source_value
            merged_count += 1

    write_json(output_path, merged)
    return merged_count


def launch_gpu_shards(args: argparse.Namespace, input_path: Path, output_path: Path) -> None:
    gpu_ids = parse_gpu_ids(args)
    shard_paths = [
        shard_output_path(output_path, shard_id, args.num_gpus)
        for shard_id in range(args.num_gpus)
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"launching {args.num_gpus} data-parallel workers on GPUs: {','.join(gpu_ids)}"
    )
    print("each worker uses tensor_parallel_size=1 and processes one shard")

    processes: list[tuple[int, Path, subprocess.Popen[Any], Any]] = []
    for shard_id, (gpu_id, shard_path) in enumerate(zip(gpu_ids, shard_paths)):
        log_path = shard_path.with_suffix(shard_path.suffix + ".log")
        log_file = log_path.open("w", encoding="utf-8")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        cmd = build_worker_command(args, shard_id, shard_path)
        print(f"[shard {shard_id}/{args.num_gpus}] GPU {gpu_id} -> {shard_path}")
        process = subprocess.Popen(
            cmd,
            cwd=Path.cwd(),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
        )
        processes.append((shard_id, log_path, process, log_file))

    failed: list[tuple[int, Path, int]] = []
    for shard_id, log_path, process, log_file in processes:
        return_code = process.wait()
        log_file.close()
        if return_code != 0:
            failed.append((shard_id, log_path, return_code))

    if failed:
        for shard_id, log_path, return_code in failed:
            print(
                f"[shard {shard_id}] failed with code {return_code}; log: {log_path}",
                file=sys.stderr,
            )
        raise SystemExit("One or more GPU shard workers failed.")

    merged_count = merge_shard_outputs(input_path, output_path, shard_paths, args)
    print(f"merged captioned samples: {merged_count}")
    print(f"wrote: {output_path}")


def apply_captions(
    data: list[dict[str, Any]],
    captions: Iterable[tuple[int, str]],
    args: argparse.Namespace,
    output_path: Path,
) -> int:
    written = 0
    for count, (index, caption) in enumerate(captions, start=1):
        message = assistant_message(data[index])
        message["value"] = insert_caption(
            str(message.get("value", "")),
            caption,
            args.caption_prefix,
        )
        written += 1
        if args.write_every > 0 and count % args.write_every == 0:
            write_json(output_path, data)
    write_json(output_path, data)
    return written


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if output_path.exists() and not args.resume and not args.force and not args.check_only:
        raise SystemExit(
            f"{output_path} already exists. Use --resume to continue or --force to overwrite."
        )

    if args.num_gpus > 1 and args.num_shards == 1 and not args.check_only:
        launch_gpu_shards(args, input_path, output_path)
        return

    print(f"loading: {input_path}")
    if args.resume and output_path.exists():
        data = load_json(output_path)
    else:
        data = load_json(input_path)

    roots = iter_video_roots(args)

    print(f"input samples: {len(data)}")
    if args.check_only:
        pending_count = sum(1 for _ in iter_pending_samples(data, args, roots))
        print(f"pending captions: {pending_count}")
        return

    pending_iter = iter_pending_samples(data, args, roots)
    try:
        first_pending = next(pending_iter)
    except StopIteration:
        write_json(output_path, data)
        print(f"nothing to annotate; wrote {output_path}")
        return
    pending_items = prepend_item(first_pending, pending_iter)

    if args.backend == "hf":
        captions = generate_hf_captions(pending_items, args)
    else:
        captions = generate_vllm_captions(pending_items, args)

    added = apply_captions(data, captions, args, output_path)
    print(f"added captions: {added}")
    print(f"wrote: {output_path}")


if __name__ == "__main__":
    main()
