#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import base64
import argparse
import subprocess
import tempfile
import shutil
import time
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI

SYS_QA = (
    "You are a video understanding assistant. Based on the user’s question, "
    "answer according to the video content and strictly follow the required output format specified by the user."
)

# -------------------------
# Utils
# -------------------------
def ensure_dir(p: str) -> None:
    if p:
        os.makedirs(p, exist_ok=True)

def b64_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def ffprobe_duration(video_path: str) -> float:
    cmd = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=nw=1:nk=1", video_path]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0

# -------------------------
# Frame extraction (single ffmpeg call)
# -------------------------
def extract_frames(
    video_path: str,
    out_dir: str,
    max_frames: int = 96,
    fps: float = 1.0,
    image_h: int = 480,
) -> List[str]:
    """
    Extract up to max_frames frames using one ffmpeg call.
    Strategy: sample at fps, but cap to max_frames via -frames:v.
    """
    ensure_dir(out_dir)
    out_pattern = os.path.join(out_dir, "frame_%06d.jpg")

    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps={fps},scale=-2:{image_h}",
        "-vsync", "0",
        "-q:v", "2",
        "-frames:v", str(int(max_frames)),
        "-y", out_pattern,
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        return []

    files = sorted(
        os.path.join(out_dir, fn) for fn in os.listdir(out_dir)
        if fn.lower().endswith(".jpg")
    )
    return files[:max_frames]

# -------------------------
# Call GPT with multi-images
# -------------------------
def call_qa(
    client: AzureOpenAI,
    model: str,
    frames: List[str],
    question: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
) -> Tuple[str, float]:
    content: List[Dict[str, Any]] = []

    # Optional: a short hint to interpret frames as evidence
    content.append({"type": "text", "text": "You will be given sampled frames from a video. Use them to answer."})

    for p in frames:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_file(p)}"}})

    content.append({"type": "text", "text": question.strip()})

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
    )
    t1 = time.time()
    ans = resp.choices[0].message.content or ""
    return ans, (t1 - t0)

# -------------------------
# Per-sample
# -------------------------
def process_one(sample: Dict[str, Any], args, client: AzureOpenAI) -> Dict[str, Any]:
    out = {
        "id": sample.get("id", sample.get("question_id", None)),
        "video_path": sample.get("video_path"),
        "question": sample.get("question"),
        "answer": None,
        "latency_sec": None,
        "error": None,
    }

    try:
        video_path = sample["video_path"]
        question = sample["question"]
        abs_video = video_path if os.path.isabs(video_path) else os.path.join(args.base_video_dir, video_path)

        if not os.path.exists(abs_video):
            raise FileNotFoundError(f"video not found: {abs_video}")

        # temp dir per sample (auto cleaned)
        tmpdir = tempfile.mkdtemp(prefix="gpt_frames_")
        try:
            frames = extract_frames(
                video_path=abs_video,
                out_dir=tmpdir,
                max_frames=args.max_frames,
                fps=args.fps,
                image_h=args.image_h,
            )
            if not frames:
                raise RuntimeError("ffmpeg frame extraction failed (no frames).")

            ans, latency = call_qa(
                client=client,
                model=args.model_name,
                frames=frames,
                question=question,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        out["answer"] = ans
        out["latency_sec"] = round(latency, 4)
        return out

    except Exception as e:
        out["error"] = f"{type(e).__name__}: {e}"
        return out

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser("minimal_video_frame_qa_batch")

    ap.add_argument("--input_json", type=str, required=True, help="list of samples: {video_path, question, (optional) id}")
    ap.add_argument("--base_video_dir", type=str, default=".", help="prefix for relative video_path")
    ap.add_argument("--output_json", type=str, required=True)

    ap.add_argument("--max_workers", type=int, default=8)

    # frame sampling
    ap.add_argument("--max_frames", type=int, default=96)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--image_h", type=int, default=480)

    # model
    ap.add_argument("--azure_endpoint", type=str, required=True)
    ap.add_argument("--azure_api_key", type=str, required=True)
    ap.add_argument("--azure_api_version", type=str, default="2024-03-01-preview")
    ap.add_argument("--model_name", type=str, required=True)

    ap.add_argument("--max_tokens", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=0.0)

    args = ap.parse_args()
    ensure_dir(os.path.dirname(args.output_json))

    with open(args.input_json, "r", encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        raise ValueError("input_json must be a JSON list.")

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.azure_api_key,
        api_version=args.azure_api_version,
    )

    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = [ex.submit(process_one, s, args, client) for s in samples]
        for fut in as_completed(futs):
            results.append(fut.result())

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Saved: {args.output_json}  (n={len(results)})")

if __name__ == "__main__":
    main()