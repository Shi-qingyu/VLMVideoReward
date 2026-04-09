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
import copy
import re
from typing import Any, Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import AzureOpenAI
from tqdm import tqdm


SYS_PROMPT = (
    "You are a video anomaly timestamp annotation assistant. "
    "You will be given sampled frames from a video, and each frame is associated with a normalized timestamp between 0.0 and 1.0. "
    "Your task is to identify the anomaly timestamp for each provided noun based on the video content. "
    "You must strictly follow the user's output format requirements."
)

USER_TEMPLATE = (
    "Identify the exact timestamps when the following nouns exhibit anomalous behavior in the video: {nouns}. "
    "Provide the results as normalized values between 0.0 and 1.0, where 0.0 is the start and 1.0 is the end of the video. "
    "Ensure that the number of output timestamps exactly matches the number of {num_noun}, and the order of timestamps strictly matches the order of the nouns. "
    "Return ONLY the numerical values separated by commas, with no additional text."
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
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=nw=1:nk=1",
        video_path
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    try:
        return float(r.stdout.strip())
    except Exception:
        return 0.0


# -------------------------
# Frame extraction
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
        os.path.join(out_dir, fn)
        for fn in os.listdir(out_dir)
        if fn.lower().endswith(".jpg")
    )
    return files[:max_frames]


def build_frame_time_map(frames: List[str]) -> List[float]:
    """
    Since frames are uniformly sampled in temporal order,
    assign normalized timestamps based on frame index.
    """
    n = len(frames)
    if n == 0:
        return []
    if n == 1:
        return [0.0]
    return [i / (n - 1) for i in range(n)]


# -------------------------
# Output parsing
# -------------------------
def parse_timestamps(output_text: str, expected_count: int) -> List[float]:
    """
    Parse comma-separated floats from model output.
    Be tolerant to spaces/newlines, but strict on final count.
    """
    if not output_text:
        return []

    text = output_text.strip()
    # 提取所有浮点数/整数，包括负号
    nums = re.findall(r"-?\d+(?:\.\d+)?", text)
    values = []
    for x in nums:
        try:
            values.append(float(x))
        except Exception:
            pass

    if len(values) != expected_count:
        return []

    # 可选：截断到合理范围，保留 -1.0
    normed = []
    for v in values:
        if v == -1.0:
            normed.append(-1.0)
        else:
            normed.append(min(1.0, max(0.0, v)))
    return normed


# -------------------------
# GPT call
# -------------------------
def call_gpt_for_timestamps(
    client: AzureOpenAI,
    model: str,
    frames: List[str],
    nouns: str,
    num_noun: int,
    max_tokens: int = 256,
    temperature: float = 0.0,
) -> Tuple[str, float]:
    """
    Send sampled frames with per-frame normalized timestamp hints.
    """
    question = USER_TEMPLATE.format(nouns=nouns, num_noun=num_noun)
    time_marks = build_frame_time_map(frames)

    content: List[Dict[str, Any]] = []
    content.append({
        "type": "text",
        "text": (
            "You will be given sampled frames from a video in chronological order. "
            "Each frame is annotated with its normalized timestamp. "
            "Use the frames to infer anomaly timestamps."
        )
    })

    for i, (frame_path, ts) in enumerate(zip(frames, time_marks)):
        content.append({
            "type": "text",
            "text": f"Frame {i + 1}, normalized timestamp: {ts:.4f}"
        })
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64_file(frame_path)}"
            }
        })

    content.append({
        "type": "text",
        "text": question
    })

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        stream=False,
    )

    t1 = time.time()

    ans = resp.choices[0].message.content or ""
    return ans, (t1 - t0)


# -------------------------
# Per sample
# -------------------------
def process_one(sample: Dict[str, Any], args, client: AzureOpenAI) -> Dict[str, Any]:
    res_entry = copy.deepcopy(sample)
    res_entry["timestamps"] = []
    res_entry["raw_output"] = None
    res_entry["latency_sec"] = None
    res_entry["error"] = None

    try:
        nouns = sample["nouns_raw"]
        nouns_list = sample["nouns"]
        num_noun = len(nouns_list)
        video_rel = sample["video"]
        video_path = video_rel if os.path.isabs(video_rel) else os.path.join(args.base_video_dir, video_rel)

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"video not found: {video_path}")

        tmpdir = tempfile.mkdtemp(prefix="gpt_frames_")
        try:
            frames = extract_frames(
                video_path=video_path,
                out_dir=tmpdir,
                max_frames=args.max_frames,
                fps=args.fps,
                image_h=args.image_h,
            )
            if not frames:
                raise RuntimeError("ffmpeg frame extraction failed (no frames).")

            raw_output, latency = call_gpt_for_timestamps(
                client=client,
                model=args.model_name,
                frames=frames,
                nouns=nouns,
                num_noun=num_noun,
                max_tokens=args.max_tokens,
            )
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

        expected_count = len([x for x in str(nouns).split(",") if x.strip()])
        parsed = parse_timestamps(raw_output, expected_count)

        res_entry["timestamps"] = parsed if parsed else raw_output
        res_entry["raw_output"] = raw_output
        res_entry["latency_sec"] = round(latency, 4)

        if not parsed:
            res_entry["error"] = (
                f"parse_failed: expected {expected_count} timestamps, got raw output: {raw_output}"
            )

        return res_entry

    except Exception as e:
        res_entry["error"] = f"{type(e).__name__}: {e}"
        return res_entry


# -------------------------
# Main
# -------------------------
def load_data(json_file: str):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Data file not found: {json_file}")
    with open(json_file, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser("gpt5_video_anomaly_timestamp_annotation")

    parser.add_argument("--data_file", type=str, default="data/train_nouns.json")
    parser.add_argument("--base_video_dir", type=str, default="data")
    parser.add_argument("--output_file", type=str, default="annotated_results_gpt5.json")

    parser.add_argument("--max_workers", type=int, default=4)

    # frame sampling
    parser.add_argument("--max_frames", type=int, default=96)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--image_h", type=int, default=480)

    # Azure OpenAI
    parser.add_argument("--azure_endpoint", type=str, default="")
    parser.add_argument("--azure_api_key", type=str, default="")
    parser.add_argument("--azure_api_version", type=str, default="")
    parser.add_argument("--model_name", type=str, default="")

    parser.add_argument("--max_tokens", type=int, default=256)

    args = parser.parse_args()
    ensure_dir(os.path.dirname(args.output_file))

    examples = load_data(args.data_file)
    if not isinstance(examples, list):
        raise ValueError("data_file must be a JSON list.")

    total_num = len(examples)

    # 读取已有结果
    if os.path.exists(args.output_file):
        try:
            with open(args.output_file, "r", encoding="utf-8") as f:
                results = json.load(f)
            if not isinstance(results, list):
                print("[Warning] Existing output file is not a JSON list. Reinitialize.")
                results = [None] * total_num
        except Exception as e:
            print(f"[Warning] Failed to load existing output file: {e}")
            print("[Warning] Reinitialize output buffer.")
            results = [None] * total_num
    else:
        results = [None] * total_num

    # 长度对齐
    if len(results) < total_num:
        results.extend([None] * (total_num - len(results)))
    elif len(results) > total_num:
        print(f"[Warning] output_file has more entries than data_file ({len(results)} > {total_num}), truncating.")
        results = results[:total_num]

    # 找出未完成 index
    unfinished_indices = [i for i, x in enumerate(results) if x is None]

    finished_num = total_num - len(unfinished_indices)
    print(f"[Resume] Finished: {finished_num} / {total_num}")
    print(f"[Resume] Remaining: {len(unfinished_indices)}")

    if not unfinished_indices:
        print("All samples already processed.")
        return

    client = AzureOpenAI(
        azure_endpoint=args.azure_endpoint,
        api_key=args.azure_api_key,
        api_version=args.azure_api_version,
    )

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        future_to_idx = {
            ex.submit(process_one, examples[idx], args, client): idx
            for idx in unfinished_indices
        }

        for fut in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="Annotating Videos"):
            idx = future_to_idx[fut]

            try:
                result = fut.result()
            except Exception as e:
                result = {
                    "error": f"FutureError: {type(e).__name__}: {e}"
                }

            results[idx] = result

            with open(args.output_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    with open(args.output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Done! Results saved to {args.output_file}")



if __name__ == "__main__":
    main()
