#!/usr/bin/env python3
"""Annotate time-tagged physics violations with SAM3 bboxes.

For every assistant ``<think>`` span containing ``<t>start-end</t>``, this
script samples the video frame at the start and end timestamps, prompts SAM3
with the visible anomalous subject/object inferred from the evidence text, and
replaces the time tag with point-time bbox annotations:

    <t>0.5s</t><box>[x1,y1,x2,y2]</box> to <t>1.5s</t><box>[x1,y1,x2,y2]</box>

Boxes are normalized to the 0..1000 xyxy format used by the existing bbox
visualization and training utilities in this repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_INPUT_PATH = "data/physics_violation_sampled_train.json"
DEFAULT_OUTPUT_PATH = "data/physics_violation_sampled_train_sam3_time_bbox.json"
DEFAULT_WORK_DIR = "data/sam3_time_bbox_shards"

TIME_RANGE_PATTERN = re.compile(
    r"<t>\s*([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\s*</t>",
    flags=re.IGNORECASE,
)
THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", flags=re.DOTALL | re.IGNORECASE)
TAG_PATTERN = re.compile(r"<[^>]+>")

LEADING_EVIDENCE_PREFIX = re.compile(
    r"^\s*(?:video caption\s*:\s*.*?(?:\n|$))*"
    r"(?:physics violation issue|issue|violation|evidence)\s*:\s*",
    flags=re.IGNORECASE | re.DOTALL,
)

BAD_OBJECT_PROMPTS = {
    "scene",
    "the scene",
    "issue",
    "violation",
    "motion",
    "movement",
    "movements",
    "expressions",
    "interaction",
    "interactions",
    "physics",
    "physical laws",
}

OBJECT_STOP_PATTERN = re.compile(
    r"\b("
    r"suddenly|stands?|moves?|moving|releases?|floats?|floating|rises?|rising|"
    r"falls?|falling|disconnects?|disappears?|appears?|remains?|shakes?|shaking|"
    r"deforms?|changes?|rotates?|slides?|jumps?|flies?|flying|levitates?|"
    r"will|would|does|do|did|can|is|are|was|were|looks?|seems?|becomes?|"
    r"unzipped?|opens?|closes?|turns?|transforms?|interacts?"
    r")\b",
    flags=re.IGNORECASE,
)


@dataclass
class TimeSpan:
    match_start: int
    match_end: int
    start_sec: float
    end_sec: float
    evidence: str


@dataclass
class BBoxResult:
    sample: dict[str, Any]
    changed: bool
    annotated_spans: int
    skipped_spans: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Insert SAM3 start/end bbox annotations for <t> ranges in <think>.",
    )
    parser.add_argument("--input", default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--work-dir", default=DEFAULT_WORK_DIR)
    parser.add_argument(
        "--video-root",
        action="append",
        default=None,
        help="Root joined with relative video paths. Can be repeated. Defaults to cwd/data.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["heuristic", "evidence"],
        default="heuristic",
        help=(
            "How to create the SAM3 text prompt. heuristic extracts a likely "
            "object phrase from the evidence; evidence uses the cleaned evidence text."
        ),
    )
    parser.add_argument("--max-prompt-words", type=int, default=14)
    parser.add_argument("--min-score", type=float, default=0.0)
    parser.add_argument(
        "--prefer-mask-box",
        action="store_true",
        default=True,
        help="Prefer bbox computed from SAM3 masks when masks are available.",
    )
    parser.add_argument("--no-prefer-mask-box", dest="prefer_mask_box", action="store_false")
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Insert a time annotation when only one of start/end boxes is found.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--write-every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge torchrun shard outputs from --work-dir into --output.",
    )
    parser.add_argument(
        "--device-index",
        type=int,
        default=None,
        help="CUDA device index for single-process runs. Defaults to LOCAL_RANK or 0.",
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


def append_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_done_set(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            key = line.strip()
            if key:
                done.add(key)
    return done


def append_done(path: Path, key: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(key + "\n")
        f.flush()
        os.fsync(f.fileno())


def sample_key(sample: dict[str, Any], index: int) -> str:
    metadata = sample.get("metadata")
    if isinstance(metadata, dict):
        for key_name in ("id", "source_line", "video_url"):
            value = metadata.get(key_name)
            if value is not None:
                return f"{index}:{key_name}:{value}"

    raw = json.dumps(
        {
            "index": index,
            "videos": sample.get("videos"),
            "assistant": assistant_message(sample).get("value", ""),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def normalize_media_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def iter_video_roots(args: argparse.Namespace) -> list[Path]:
    roots = args.video_root or ["data"]
    return [Path(root).expanduser().resolve() for root in roots]


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
        metadata = sample.get("metadata")
        if isinstance(metadata, dict) and metadata.get("video_path"):
            videos = [str(metadata["video_path"])]
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


def answer_is_yes(value: str) -> bool:
    match = ANSWER_PATTERN.search(value)
    if not match:
        return False
    answer = re.sub(r"\s+", " ", match.group(1)).strip().lower()
    return answer == "yes"


def clean_evidence(text: str) -> str:
    text = re.sub(r"<box>.*?</box>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = TAG_PATTERN.sub(" ", text)
    text = LEADING_EVIDENCE_PREFIX.sub("", text)
    text = re.sub(r"\s+", " ", text).strip(" .,:;\"'")
    return text


def first_sentence(text: str) -> str:
    match = re.search(r"(.+?[。！？.!?])(?:\s|$)", text)
    if match:
        return match.group(1).strip(" .,:;\"'")
    return text.splitlines()[0].strip(" .,:;\"'") if text.splitlines() else text


def limit_words(text: str, max_words: int) -> str:
    words = text.split()
    if max_words <= 0 or len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def normalize_object_prompt(prompt: str, max_words: int) -> str | None:
    prompt = prompt.strip(" .,:;\"'")
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = re.sub(r"^(?:the|a|an|this|that)\s+", "", prompt, flags=re.IGNORECASE)
    prompt = re.sub(r"\b(?:that|which|who)$", "", prompt, flags=re.IGNORECASE).strip()
    prompt = re.sub(r"'s\b.*$", "", prompt).strip()
    prompt = prompt.strip(" .,:;\"'")
    prompt = limit_words(prompt, max_words)
    lower_prompt = prompt.lower()
    if not prompt or lower_prompt in BAD_OBJECT_PROMPTS:
        return None
    if any(lower_prompt.startswith(f"{bad} ") for bad in BAD_OBJECT_PROMPTS):
        return None
    return prompt


def heuristic_object_prompt(evidence: str, max_words: int) -> str | None:
    text = first_sentence(clean_evidence(evidence))
    if not text:
        return None

    between_match = re.search(
        r"\bbetween\s+(.+?)\s+and\s+(.+?)(?:[.,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if between_match:
        right = normalize_object_prompt(between_match.group(2), max_words)
        if right:
            return right

    article_match = re.search(
        r"^(?:the|a|an|this|that)\s+(.+?)\s+" + OBJECT_STOP_PATTERN.pattern,
        text,
        flags=re.IGNORECASE,
    )
    if article_match:
        prompt = normalize_object_prompt(article_match.group(1), max_words)
        if prompt:
            return prompt

    stop_match = OBJECT_STOP_PATTERN.search(text)
    if stop_match:
        prompt = normalize_object_prompt(text[: stop_match.start()], max_words)
        if prompt:
            return prompt

    return normalize_object_prompt(text, max_words)


def object_prompt_for_span(evidence: str, args: argparse.Namespace) -> str | None:
    if args.prompt_mode == "evidence":
        return normalize_object_prompt(
            first_sentence(clean_evidence(evidence)),
            args.max_prompt_words,
        )
    return heuristic_object_prompt(evidence, args.max_prompt_words)


def collect_time_spans(value: str) -> list[TimeSpan]:
    spans: list[TimeSpan] = []
    for think_match in THINK_PATTERN.finditer(value):
        think_start = think_match.start(1)
        think_end = think_match.end(1)
        time_matches = list(TIME_RANGE_PATTERN.finditer(value, think_start, think_end))
        for idx, match in enumerate(time_matches):
            evidence_start = match.end()
            evidence_end = time_matches[idx + 1].start() if idx + 1 < len(time_matches) else think_end
            evidence = value[evidence_start:evidence_end]
            spans.append(
                TimeSpan(
                    match_start=match.start(),
                    match_end=match.end(),
                    start_sec=float(match.group(1)),
                    end_sec=float(match.group(2)),
                    evidence=evidence,
                )
            )
    return spans


def read_video_meta(video_path: Path) -> tuple[int, float]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return 0, 0.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return num_frames, fps


def seconds_to_frame_index(seconds: float, fps: float, num_frames: int) -> int:
    if num_frames <= 0:
        return 0
    frame_index = int(seconds * fps)
    return max(0, min(frame_index, num_frames - 1))


def read_frame_as_pil(video_path: Path, frame_index: int) -> Any | None:
    import cv2
    from PIL import Image

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        return None
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)


def tensor_or_array_len(value: Any) -> int:
    if value is None:
        return 0
    try:
        return len(value)
    except TypeError:
        return 0


def get_item(value: Any, index: int) -> Any:
    import numpy as np
    import torch

    if isinstance(value, torch.Tensor):
        return value[index]
    if isinstance(value, np.ndarray):
        return value[index]
    return value[index]


def score_to_float(score: Any) -> float:
    import numpy as np
    import torch

    if score is None:
        return 0.0
    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().reshape(-1)[0])
    if isinstance(score, np.ndarray):
        return float(score.reshape(-1)[0])
    return float(score)


def clamp_box_1000(box: Iterable[float]) -> list[int]:
    return [int(max(0, min(1000, round(float(value))))) for value in box]


def box_xyxy_to_1000(box: Any, image_w: int, image_h: int) -> list[int] | None:
    import numpy as np

    values = np.asarray(box, dtype=np.float32).reshape(-1).tolist()
    if len(values) < 4:
        return None
    x1, y1, x2, y2 = values[:4]
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        return clamp_box_1000([x1 * 1000, y1 * 1000, x2 * 1000, y2 * 1000])
    return clamp_box_1000(
        [
            x1 / image_w * 1000,
            y1 / image_h * 1000,
            x2 / image_w * 1000,
            y2 / image_h * 1000,
        ]
    )


def mask_to_box_xyxy_1000(mask: Any) -> list[int] | None:
    import numpy as np
    import torch

    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()
    mask = np.asarray(mask)
    mask = np.squeeze(mask)
    if mask.ndim != 2:
        return None

    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        return None

    height, width = mask.shape
    return clamp_box_1000(
        [
            xs.min() / width * 1000,
            ys.min() / height * 1000,
            (xs.max() + 1) / width * 1000,
            (ys.max() + 1) / height * 1000,
        ]
    )


def build_sam3_image_processor(device_index: int):
    import torch

    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    if torch.cuda.is_available():
        torch.cuda.set_device(device_index)
        device = torch.device(f"cuda:{device_index}")
    else:
        device = torch.device("cpu")

    model = build_sam3_image_model()
    try:
        model = model.to(device)
    except Exception:
        pass
    try:
        model.eval()
    except Exception:
        pass
    return Sam3Processor(model)


def sam3_prompt_to_box(
    processor,
    image: Any,
    text_prompt: str,
    min_score: float,
    prefer_mask_box: bool,
) -> list[int] | None:
    import torch

    image_w, image_h = image.size
    with torch.inference_mode():
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=text_prompt)

    masks = output.get("masks")
    boxes = output.get("boxes")
    scores = output.get("scores")

    n_masks = tensor_or_array_len(masks)
    n_boxes = tensor_or_array_len(boxes)
    n_scores = tensor_or_array_len(scores)
    instance_count = max(n_masks, n_boxes, n_scores)
    if instance_count <= 0:
        return None

    best_idx = 0
    if n_scores > 0:
        best_score = -1e9
        for idx in range(n_scores):
            score = score_to_float(get_item(scores, idx))
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_score < min_score:
            return None

    if prefer_mask_box and n_masks > best_idx:
        box = mask_to_box_xyxy_1000(get_item(masks, best_idx))
        if box is not None:
            return box

    if n_boxes > best_idx:
        box = box_xyxy_to_1000(get_item(boxes, best_idx), image_w, image_h)
        if box is not None:
            return box

    if n_masks > best_idx:
        box = mask_to_box_xyxy_1000(get_item(masks, best_idx))
        if box is not None:
            return box

    return None


def format_seconds(seconds: float) -> str:
    if abs(seconds - round(seconds)) < 1e-6:
        return f"{int(round(seconds))}s"
    return f"{seconds:.1f}s"


def format_box(box: list[int]) -> str:
    return json.dumps(box, separators=(",", ":"))


def format_time_annotation(
    span: TimeSpan,
    start_box: list[int] | None,
    end_box: list[int] | None,
) -> str:
    start = f"<t>{format_seconds(span.start_sec)}</t>"
    end = f"<t>{format_seconds(span.end_sec)}</t>"
    if start_box is not None:
        start += f"<box>{format_box(start_box)}</box>"
    if end_box is not None:
        end += f"<box>{format_box(end_box)}</box>"
    return f"{start} to {end}"


def apply_replacements(value: str, replacements: list[tuple[int, int, str]]) -> str:
    for start, end, replacement in sorted(replacements, reverse=True):
        value = value[:start] + replacement + value[end:]
    return value


def annotate_sample(
    sample: dict[str, Any],
    video_path: Path,
    processor,
    args: argparse.Namespace,
) -> BBoxResult:
    output = copy.deepcopy(sample)
    message = assistant_message(output)
    value = str(message.get("value", ""))

    if not answer_is_yes(value):
        return BBoxResult(output, changed=False, annotated_spans=0, skipped_spans=0)

    spans = collect_time_spans(value)
    if not spans:
        return BBoxResult(output, changed=False, annotated_spans=0, skipped_spans=0)

    num_frames, fps = read_video_meta(video_path)
    if num_frames <= 0 or fps <= 0:
        return BBoxResult(output, changed=False, annotated_spans=0, skipped_spans=len(spans))

    frame_cache: dict[int, Any | None] = {}
    replacements: list[tuple[int, int, str]] = []
    annotated_spans = 0
    skipped_spans = 0

    for span in spans:
        text_prompt = object_prompt_for_span(span.evidence, args)
        if not text_prompt:
            skipped_spans += 1
            continue

        frame_indices = {
            "start": seconds_to_frame_index(span.start_sec, fps, num_frames),
            "end": seconds_to_frame_index(span.end_sec, fps, num_frames),
        }
        boxes: dict[str, list[int] | None] = {"start": None, "end": None}

        for name, frame_index in frame_indices.items():
            if frame_index not in frame_cache:
                frame_cache[frame_index] = read_frame_as_pil(video_path, frame_index)
            image = frame_cache[frame_index]
            if image is None:
                continue
            try:
                boxes[name] = sam3_prompt_to_box(
                    processor=processor,
                    image=image,
                    text_prompt=text_prompt,
                    min_score=args.min_score,
                    prefer_mask_box=args.prefer_mask_box,
                )
            except Exception:
                boxes[name] = None

        start_box = boxes["start"]
        end_box = boxes["end"]
        has_both = start_box is not None and end_box is not None
        has_any = start_box is not None or end_box is not None
        if not has_both and not (args.allow_partial and has_any):
            skipped_spans += 1
            continue

        replacements.append(
            (
                span.match_start,
                span.match_end,
                format_time_annotation(span, start_box, end_box),
            )
        )
        annotated_spans += 1

    if replacements:
        message["value"] = apply_replacements(value, replacements)

    return BBoxResult(
        output,
        changed=bool(replacements),
        annotated_spans=annotated_spans,
        skipped_spans=skipped_spans,
    )


def selected_indices(total: int, start_index: int, limit: int | None) -> range:
    start = max(0, start_index)
    stop = total if limit is None else min(total, start + max(0, limit))
    return range(start, stop)


def iter_pending_indices(
    data: list[dict[str, Any]],
    args: argparse.Namespace,
    roots: list[Path],
) -> Iterable[tuple[int, Path, int]]:
    missing_count = 0
    for index in selected_indices(len(data), args.start_index, args.limit):
        sample = data[index]
        try:
            value = str(assistant_message(sample).get("value", ""))
            pending_spans = len(collect_time_spans(value)) if answer_is_yes(value) else 0
            if pending_spans <= 0:
                continue
            yield index, resolve_video_path(sample, roots), pending_spans
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


def check_dataset(
    data: list[dict[str, Any]],
    args: argparse.Namespace,
    roots: list[Path],
) -> None:
    pending_samples = 0
    pending_spans = 0
    missing_count = 0

    for index in selected_indices(len(data), args.start_index, args.limit):
        sample = data[index]
        try:
            value = str(assistant_message(sample).get("value", ""))
            span_count = len(collect_time_spans(value)) if answer_is_yes(value) else 0
            if span_count <= 0:
                continue
            pending_samples += 1
            pending_spans += span_count
            resolve_video_path(sample, roots)
        except Exception as exc:
            missing_count += 1
            if missing_count <= 10:
                print(f"[missing] index={index}: {exc}", file=sys.stderr)
            elif missing_count == 11:
                print("[missing] further missing/invalid samples omitted", file=sys.stderr)

    if missing_count:
        print(f"missing/invalid pending samples: {missing_count}", file=sys.stderr)
    print(f"pending samples: {pending_samples}")
    print(f"pending time spans: {pending_spans}")


def progress(items: Iterable[Any], desc: str = "", total: int | None = None) -> Iterable[Any]:
    try:
        from tqdm import tqdm
    except ImportError:
        return items
    return tqdm(items, desc=desc, total=total)


def device_index_from_env(args: argparse.Namespace) -> int:
    if args.device_index is not None:
        return args.device_index
    return int(os.environ.get("LOCAL_RANK", "0"))


def world_info() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    return rank, local_rank, world_size


def shard_paths(work_dir: Path, rank: int) -> tuple[Path, Path, Path]:
    return (
        work_dir / f"sam3_time_bbox_rank{rank}.jsonl",
        work_dir / f"sam3_time_bbox_rank{rank}.done",
        work_dir / f"sam3_time_bbox_rank{rank}.errors.jsonl",
    )


def run_single_process(args: argparse.Namespace, input_path: Path, output_path: Path) -> None:
    if output_path.exists() and not args.resume and not args.force and not args.check_only:
        raise SystemExit(
            f"{output_path} already exists. Use --resume to continue or --force to overwrite."
        )

    print(f"loading: {input_path}")
    data = load_json(output_path if args.resume and output_path.exists() else input_path)
    roots = iter_video_roots(args)
    print(f"input samples: {len(data)}")
    if args.check_only:
        check_dataset(data, args, roots)
        return

    pending = list(iter_pending_indices(data, args, roots))
    pending_spans = sum(item[2] for item in pending)

    print(f"pending samples: {len(pending)}")
    print(f"pending time spans: {pending_spans}")
    if not pending:
        write_json(output_path, data)
        print(f"nothing to annotate; wrote {output_path}")
        return

    processor = build_sam3_image_processor(device_index_from_env(args))
    changed = 0
    annotated_spans = 0
    skipped_spans = 0

    for count, (index, video_path, _) in enumerate(
        progress(pending, desc="SAM3 bbox", total=len(pending)),
        start=1,
    ):
        result = annotate_sample(data[index], video_path, processor, args)
        data[index] = result.sample
        changed += int(result.changed)
        annotated_spans += result.annotated_spans
        skipped_spans += result.skipped_spans
        if args.write_every > 0 and count % args.write_every == 0:
            write_json(output_path, data)

    write_json(output_path, data)
    print(f"changed samples: {changed}")
    print(f"annotated time spans: {annotated_spans}")
    print(f"skipped time spans: {skipped_spans}")
    print(f"wrote: {output_path}")


def run_shard_worker(
    args: argparse.Namespace,
    input_path: Path,
    rank: int,
    local_rank: int,
    world_size: int,
) -> None:
    data = load_json(input_path)
    roots = iter_video_roots(args)
    work_dir = Path(args.work_dir)
    jsonl_path, done_path, err_path = shard_paths(work_dir, rank)

    if args.force:
        for path in (jsonl_path, done_path, err_path):
            if path.exists():
                path.unlink()
    done = set() if args.force else load_done_set(done_path)
    processor = build_sam3_image_processor(args.device_index if args.device_index is not None else local_rank)

    indices = list(selected_indices(len(data), args.start_index, args.limit))
    indices = [index for index in indices if index % world_size == rank]

    changed = 0
    annotated_spans = 0
    skipped_spans = 0
    for index in progress(indices, desc=f"SAM3 bbox rank {rank}", total=len(indices)):
        try:
            key = sample_key(data[index], index)
            if key in done:
                continue
            value = str(assistant_message(data[index]).get("value", ""))
            if not answer_is_yes(value) or not collect_time_spans(value):
                append_jsonl(jsonl_path, {"index": index, "sample": data[index]})
                append_done(done_path, key)
                done.add(key)
                continue
            video_path = resolve_video_path(data[index], roots)
            result = annotate_sample(data[index], video_path, processor, args)
            append_jsonl(jsonl_path, {"index": index, "sample": result.sample})
            append_done(done_path, key)
            done.add(key)
            changed += int(result.changed)
            annotated_spans += result.annotated_spans
            skipped_spans += result.skipped_spans
        except Exception as exc:
            append_jsonl(
                err_path,
                {
                    "index": index,
                    "video": data[index].get("videos"),
                    "error": str(exc),
                },
            )

    print(
        f"[rank {rank}] changed={changed} annotated_spans={annotated_spans} "
        f"skipped_spans={skipped_spans}"
    )


def merge_shards(input_path: Path, output_path: Path, work_dir: Path) -> int:
    merged = load_json(input_path)
    applied = 0
    for shard_path in sorted(work_dir.glob("sam3_time_bbox_rank*.jsonl")):
        if shard_path.name.endswith(".errors.jsonl"):
            continue
        with shard_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                index = int(item["index"])
                merged[index] = item["sample"]
                applied += 1
    write_json(output_path, merged)
    return applied


def maybe_init_distributed(world_size: int):
    if world_size <= 1:
        return None
    import torch
    import torch.distributed as dist

    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
    return dist


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    work_dir = Path(args.work_dir)
    rank, local_rank, world_size = world_info()

    if args.merge_only:
        applied = merge_shards(input_path, output_path, work_dir)
        print(f"merged shard items: {applied}")
        print(f"wrote: {output_path}")
        return

    if world_size > 1:
        dist = maybe_init_distributed(world_size)
        run_shard_worker(args, input_path, rank, local_rank, world_size)
        if dist is not None:
            dist.barrier()
        if rank == 0:
            applied = merge_shards(input_path, output_path, work_dir)
            print(f"merged shard items: {applied}")
            print(f"wrote: {output_path}")
        if dist is not None:
            dist.barrier()
        return

    run_single_process(args, input_path, output_path)


if __name__ == "__main__":
    main()
