#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import base64
import argparse
import subprocess
import tempfile
import shutil
import time
import copy
from typing import Any, Dict, List, Tuple, Optional
from multiprocessing import Process, Queue

from openai import AzureOpenAI


SYS_QA = (
    "You are a video reasoning assistant. "
    "You will be given sampled frames from a video in chronological order, and each frame includes its timestamp. "
    "Answer strictly in the format requested by the user."
)


# -------------------------
# Basic utils
# -------------------------
def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)


def sample_key(sample: Dict[str, Any]) -> str:
    videos = sample.get("videos", [])
    if videos:
        return videos[0]
    return json.dumps(sample, ensure_ascii=False, sort_keys=True)


def append_jsonl(path: str, item: Dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    out = []
    if not os.path.exists(path):
        return out
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def load_done_set_from_jsonl(path: str) -> set:
    done = set()
    for item in load_jsonl(path):
        try:
            done.add(sample_key(item))
        except Exception:
            continue
    return done


def rewrite_json_from_items(items: List[Dict[str, Any]], json_path: str) -> None:
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


# -------------------------
# Video utils
# -------------------------
def ffprobe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def extract_frames_uniform(
    video_path: str,
    out_dir: str,
    num_frames: int = 8,
    image_h: int = 480,
) -> Tuple[List[str], List[float], float]:
    """
    全片均匀采样 num_frames 帧。
    返回: frame_paths, timestamps, duration_sec
    """
    ensure_dir(out_dir)
    duration_sec = ffprobe_duration(video_path)
    if duration_sec <= 0:
        return [], [], 0.0

    if num_frames <= 1:
        timestamps = [0.0]
    else:
        step = duration_sec / num_frames
        timestamps = [min(duration_sec, (i + 0.5) * step) for i in range(num_frames)]

    frame_paths = []
    for i, ts in enumerate(timestamps):
        out_path = os.path.join(out_dir, f"frame_{i:06d}.jpg")
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-ss", f"{ts:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-vf", f"scale=-2:{image_h}",
            "-q:v", "2",
            "-y", out_path,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0 and os.path.exists(out_path):
            frame_paths.append(out_path)
        else:
            frame_paths.append(None)

    valid_pairs = [(p, ts) for p, ts in zip(frame_paths, timestamps) if p is not None]
    if not valid_pairs:
        return [], [], duration_sec

    frame_paths = [x[0] for x in valid_pairs]
    timestamps = [x[1] for x in valid_pairs]
    return frame_paths, timestamps, duration_sec


# -------------------------
# Parsing helpers
# -------------------------
def safe_load_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        return json.loads(m.group(0))

    raise ValueError(f"Cannot parse JSON from response: {text}")


def extract_tag_content(text: str, tag_name: str) -> str:
    pattern = rf"<{tag_name}>\s*(.*?)\s*</{tag_name}>"
    m = re.search(pattern, text, flags=re.DOTALL)
    return m.group(1).strip() if m else ""


def replace_tag_content(text: str, tag_name: str, new_content: str) -> str:
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"
    start_pos = text.find(start_tag)
    end_pos = text.find(end_tag)
    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        prefix = text[: start_pos + len(start_tag)]
        suffix = text[end_pos:]
        return f"{prefix}\n{new_content}\n{suffix}"
    return text


def strip_existing_interval_prefix(text: str) -> str:
    return re.sub(r"^<t>\d+(\.\d+)?s-\d+(\.\d+)?s</t>\s*", "", text).strip()


def strip_label_prefix(text: str, label: str) -> str:
    pattern = rf"^\[{re.escape(label)}\]\s*:\s*"
    return re.sub(pattern, "", text.strip(), flags=re.IGNORECASE).strip()


def remove_all_t_tags(text: str) -> str:
    return re.sub(r"<t>\d+(\.\d+)?s-\d+(\.\d+)?s</t>", "", text)

def normalize_claim_for_skip(text: str) -> str:
    text = remove_all_t_tags(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def should_skip_claim(claim: str) -> bool:
    s = normalize_claim_for_skip(claim)
    return "good" in s


def parse_three_claims_from_think(thinking: str) -> List[Dict[str, str]]:
    """
    从 <think> 中解析：
      [Visual Quality]: ...
      [Motion & Physical Consistency]: ...
      [Prompt Alignment]: ...
    返回固定顺序 list[{"label":..., "claim":...}]
    """
    labels = [
        "Visual Quality",
        "Motion & Physical Consistency",
        "Prompt Alignment",
    ]

    pattern = re.compile(
        r"\[(Visual Quality|Motion\s*&\s*Physical Consistency|Prompt Alignment)\]\s*:\s*(.*?)(?=\n?\[(?:Visual Quality|Motion\s*&\s*Physical Consistency|Prompt Alignment)\]\s*:|\Z)",
        flags=re.DOTALL
    )

    found = {}
    for m in pattern.finditer(thinking):
        raw_label = re.sub(r"\s+", " ", m.group(1).strip())
        claim = m.group(2).strip()
        claim = strip_existing_interval_prefix(claim)
        found[raw_label] = claim

    results = []
    for label in labels:
        results.append({"label": label, "claim": found.get(label, "").strip()})
    return results


def build_new_think(
    claims: List[Dict[str, str]],
    intervals: List[Optional[Dict[str, Any]]],
    min_confidence: float = 0.0,
) -> str:
    """
    输出格式：
    [Visual Quality]: <t>...</t> claim
    [Motion & Physical Consistency]: <t>...</t> claim
    [Prompt Alignment]: <t>...</t> claim

    如果 claim == Good，则直接：
    [Visual Quality]: Good
    """
    lines = []

    for item, interval in zip(claims, intervals):
        label = item["label"]
        claim = item["claim"].strip()

        if not claim:
            lines.append(f"[{label}]:")
            continue

        if should_skip_claim(claim):
            lines.append(f"[{label}]: {claim}")
            continue

        if interval is None or interval.get("_error", False) or interval.get("confidence", 0.0) < min_confidence:
            lines.append(f"[{label}]: {claim}")
            continue

        prefix = f"<t>{interval['start_sec']:.1f}s-{interval['end_sec']:.1f}s</t>"
        lines.append(f"[{label}]: {prefix} {claim}")

    return "\n".join(lines).strip()


# -------------------------
# Model call
# -------------------------
def call_claim_interval(
    client: AzureOpenAI,
    model: str,
    frames: List[str],
    frame_timestamps: List[float],
    claim: str,
    duration_sec: float,
    max_tokens: int = 256,
    temperature: float = 0.0,
    request_timeout: float = 120.0,
) -> Tuple[Dict[str, Any], float]:
    content: List[Dict[str, Any]] = []

    intro = (
        f"Video duration: {duration_sec:.2f} seconds.\n"
        "You are given sampled frames in chronological order. "
        "Each frame is preceded by its timestamp.\n\n"
        "Find the single most relevant time interval that best supports the claim.\n\n"
        "Return ONLY valid JSON in this exact format: "
        '{"start_sec": <start_sec>, "end_sec": <end_sec>, "confidence": <confidence>}\n'
        f"Claim: {claim}"
    )
    content.append({"type": "text", "text": intro})

    for img_path, ts in zip(frames, frame_timestamps):
        content.append({"type": "text", "text": f"Timestamp: {ts:.1f}s"})
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64_file(img_path)}"},
        })

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_QA},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=False,
        timeout=request_timeout,
    )
    t1 = time.time()

    text = resp.choices[0].message.content or "{}"
    data = safe_load_json(text)

    start_sec = float(data.get("start_sec", 0.0))
    end_sec = float(data.get("end_sec", 0.0))
    confidence = float(data.get("confidence", 0.0))

    start_sec = max(0.0, min(start_sec, duration_sec))
    end_sec = max(0.0, min(end_sec, duration_sec))
    if end_sec < start_sec:
        start_sec, end_sec = end_sec, start_sec

    return {
        "start_sec": start_sec,
        "end_sec": end_sec,
        "confidence": confidence,
        "raw_text": text,
    }, (t1 - t0)


# -------------------------
# Per-sample
# -------------------------
def process_one(sample: Dict[str, Any], args, client: AzureOpenAI) -> Dict[str, Any]:
    new_sample = copy.deepcopy(sample)

    video_rel = new_sample["videos"][0]
    judgement = new_sample["conversations"][-1]["value"]
    thinking = extract_tag_content(judgement, "think")

    if not thinking:
        new_sample["_interval_error"] = "No <think> found."
        return new_sample

    claims = parse_three_claims_from_think(thinking)
    if not any(x["claim"] for x in claims):
        new_sample["_interval_error"] = "No target claims found in <think>."
        return new_sample

    abs_video = video_rel if os.path.isabs(video_rel) else os.path.join(args.base_video_dir, video_rel)
    if not os.path.exists(abs_video):
        new_sample["_interval_error"] = f"video not found: {abs_video}"
        return new_sample

    # 如果三条里全是空或者 Good，也不必抽帧
    need_model = False
    for x in claims:
        c = x["claim"].strip()
        if c and (not should_skip_claim(c)):
            need_model = True
            break

    frame_timestamps = []
    duration_sec = 0.0
    errors = []
    latencies = []
    intervals = []

    tmpdir = None
    frames = []

    try:
        if need_model:
            tmpdir = tempfile.mkdtemp(prefix="three_claim_interval_frames_")
            frames, frame_timestamps, duration_sec = extract_frames_uniform(
                video_path=abs_video,
                out_dir=tmpdir,
                num_frames=args.num_frames,
                image_h=args.image_h,
            )
            if not frames:
                new_sample["_interval_error"] = "frame extraction failed."
                return new_sample
        else:
            duration_sec = ffprobe_duration(abs_video)

        for item in claims:
            claim_text = item["claim"].strip()

            if not claim_text:
                intervals.append(None)
                continue

            if should_skip_claim(claim_text):
                intervals.append(None)
                continue

            try:
                interval, latency = call_claim_interval(
                    client=client,
                    model=args.model_name,
                    frames=frames,
                    frame_timestamps=frame_timestamps,
                    claim=claim_text,
                    duration_sec=duration_sec,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    request_timeout=args.request_timeout,
                )
                intervals.append(interval)
                latencies.append(latency)
            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                intervals.append({
                    "start_sec": 0.0,
                    "end_sec": 0.0,
                    "confidence": 0.0,
                    "raw_text": "",
                    "_error": True,
                    "_error_msg": err_msg,
                })
                errors.append({
                    "claim": claim_text,
                    "error": err_msg,
                })

            if args.sleep_sec > 0:
                time.sleep(args.sleep_sec)

    finally:
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)

    new_think = build_new_think(
        claims=claims,
        intervals=intervals,
        min_confidence=args.min_confidence,
    )
    new_judgement = replace_tag_content(judgement, "think", new_think)
    new_sample["conversations"][-1]["value"] = new_judgement
    new_sample["conversations"][0]["value"] = new_sample["conversations"][0]["value"].replace("..", ".")

    new_sample["_interval_debug"] = {
        "video_path": abs_video,
        "duration_sec": duration_sec,
        "claims": claims,
        "intervals": intervals,
        "frame_timestamps": frame_timestamps,
        "errors": errors,
        "latencies": latencies,
    }
    return new_sample


# -------------------------
# Worker split
# -------------------------
def assign_samples_to_workers(samples: List[Dict[str, Any]], num_workers: int) -> List[List[Dict[str, Any]]]:
    chunks = [[] for _ in range(num_workers)]
    for i, sample in enumerate(samples):
        chunks[i % num_workers].append(sample)
    return chunks


# -------------------------
# Worker
# -------------------------
def worker_main(
    worker_id: int,
    samples: List[Dict[str, Any]],
    args,
    output_jsonl: str,
    progress_q: Optional[Queue] = None,
):
    done_set = load_done_set_from_jsonl(output_jsonl)
    todo_samples = [s for s in samples if sample_key(s) not in done_set]

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.azure_api_key,
        api_version=args.azure_api_version,
        timeout=args.request_timeout,
        max_retries=2,
    )

    if progress_q is not None:
        progress_q.put({
            "type": "worker_init",
            "worker_id": worker_id,
            "done_existing": len(done_set),
            "todo": len(todo_samples),
            "total": len(samples),
        })

    for idx, sample in enumerate(todo_samples):
        try:
            out = process_one(sample, args, client)
        except Exception as e:
            out = copy.deepcopy(sample)
            out["_interval_error"] = f"WORKER_ERROR: {type(e).__name__}: {e}"

        append_jsonl(output_jsonl, out)

        if progress_q is not None:
            progress_q.put({
                "type": "worker_progress",
                "worker_id": worker_id,
                "done_now": idx + 1,
                "todo": len(todo_samples),
                "sample_key": sample_key(sample),
            })

    if progress_q is not None:
        progress_q.put({
            "type": "worker_done",
            "worker_id": worker_id,
        })


# -------------------------
# Merge
# -------------------------
def merge_worker_jsonls(
    worker_jsonls: List[str],
    final_json_path: str,
    preserve_input_order: bool,
    original_samples: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    merged = []
    seen = set()

    for path in worker_jsonls:
        items = load_jsonl(path)
        for item in items:
            k = sample_key(item)
            if k in seen:
                continue
            seen.add(k)
            merged.append(item)

    if preserve_input_order:
        mapping = {sample_key(x): x for x in merged}
        ordered = []
        for s in original_samples:
            k = sample_key(s)
            if k in mapping:
                ordered.append(mapping[k])
        merged = ordered

    rewrite_json_from_items(merged, final_json_path)
    return merged


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("tag_three_claim_intervals_mp_resume")

    ap.add_argument("--input_json", type=str, default="./data/train_fixed.json")
    ap.add_argument("--base_video_dir", type=str, default="./data")
    ap.add_argument("--output_json", type=str, default="./data/train_t.json")
    ap.add_argument("--worker_output_dir", type=str, default="./data/worker_jsonl")

    ap.add_argument("--num_workers", type=int, default=8)

    # frame sampling
    ap.add_argument("--num_frames", type=int, default=10)
    ap.add_argument("--image_h", type=int, default=480)

    # api
    ap.add_argument("--azure_endpoint", type=str, default="https://gpt-i18n.byteintl.net/gpt/openapi/online/multimodal/crawl")
    ap.add_argument("--azure_api_key", type=str, required=True)
    ap.add_argument("--azure_api_version", type=str, default="2025-04-01-preview")
    ap.add_argument("--model_name", type=str, default="gemini-2.5-pro-preview-05-06")

    # generation
    ap.add_argument("--max_tokens", type=int, default=4096)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--request_timeout", type=float, default=120.0)
    ap.add_argument("--sleep_sec", type=float, default=0.0)
    ap.add_argument("--min_confidence", type=float, default=0.0)

    # merge
    ap.add_argument("--preserve_input_order", action="store_true")

    args = ap.parse_args()

    ensure_dir(os.path.dirname(args.output_json))
    ensure_dir(args.worker_output_dir)

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("input_json must be a JSON list.")

    if len(samples) == 0:
        rewrite_json_from_items([], args.output_json)
        print(f"[FINAL] empty input -> {args.output_json}")
        return

    num_workers = max(1, min(args.num_workers, len(samples)))
    chunks = assign_samples_to_workers(samples, num_workers)

    worker_jsonls = [
        os.path.join(args.worker_output_dir, f"part_{i:03d}.jsonl")
        for i in range(num_workers)
    ]

    print(f"[INFO] total samples = {len(samples)}")
    print(f"[INFO] num_workers = {num_workers}")
    print(f"[INFO] worker_output_dir = {args.worker_output_dir}")
    print("[INFO] resume mode enabled")

    progress_q = Queue()
    procs = []

    for worker_id in range(num_workers):
        p = Process(
            target=worker_main,
            args=(worker_id, chunks[worker_id], args, worker_jsonls[worker_id], progress_q),
        )
        p.start()
        procs.append(p)

    worker_finished = set()
    while len(worker_finished) < num_workers:
        msg = progress_q.get()

        if msg["type"] == "worker_init":
            print(
                f"[WORKER-{msg['worker_id']}] existing_done={msg['done_existing']} "
                f"todo={msg['todo']} total_assigned={msg['total']}",
                flush=True,
            )
        elif msg["type"] == "worker_progress":
            print(
                f"[WORKER-{msg['worker_id']}] {msg['done_now']}/{msg['todo']} "
                f"{msg['sample_key']}",
                flush=True,
            )
        elif msg["type"] == "worker_done":
            worker_finished.add(msg["worker_id"])
            print(f"[WORKER-{msg['worker_id']}] done", flush=True)

    for p in procs:
        p.join()

    merged = merge_worker_jsonls(
        worker_jsonls=worker_jsonls,
        final_json_path=args.output_json,
        preserve_input_order=args.preserve_input_order,
        original_samples=samples,
    )
    print(f"[FINAL] merged {len(merged)} samples -> {args.output_json}", flush=True)


if __name__ == "__main__":
    main()