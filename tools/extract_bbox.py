import os
import json
import copy
import random
import hashlib
import argparse
import cv2
import torch
import glob
import re

from tqdm import tqdm
import numpy as np
from sam3.model_builder import build_sam3_video_predictor


TIME_SPAN_PATTERN = re.compile(
    r"<t>\s*([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\s*</t>",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="data/train_nouns.json")
    parser.add_argument("--output_dir", type=str, default="data/parallel_outputs")
    parser.add_argument("--merge_only", action="store_true", default=False, help="Only merge results without processing")
    return parser.parse_args()


def merge_results(output_dir):
    merged = []
    paths = sorted(glob.glob(os.path.join(output_dir, "train_spatial_temporal_rank*.jsonl")))
    for path in paths:
        if path.endswith(".errors.jsonl"):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    merged.append(json.loads(line))

    out_path = os.path.join(output_dir, "train_spatial_temporal.merged.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)


def split_data(data, rank, world_size):
    return data[rank::world_size]


def get_sample_key(d):
    if "key" in d:
        return str(d["key"])
    if "id" in d:
        return str(d["id"])
    raw = json.dumps(d, ensure_ascii=False, sort_keys=True)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_done_set(done_path):
    done = set()
    if os.path.exists(done_path):
        with open(done_path, "r", encoding="utf-8") as f:
            for line in f:
                key = line.strip()
                if key:
                    done.add(key)
    return done


def append_jsonl(jsonl_fp, item):
    jsonl_fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    jsonl_fp.flush()
    os.fsync(jsonl_fp.fileno())


def append_done(done_fp, key):
    done_fp.write(key + "\n")
    done_fp.flush()
    os.fsync(done_fp.fileno())


def build_predictor(rank):
    torch.cuda.set_device(rank)
    predictor = build_sam3_video_predictor(gpus_to_use=[rank])
    return predictor


def xywh_to_xyxy(box: list):
    x_min, y_min, w, h = box
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def extract_answer_block(text: str) -> str:
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1) if m else ""


def all_answers_yes(text: str) -> bool:
    answer_block = extract_answer_block(text)
    if not answer_block:
        return False
    yn = re.findall(r":\s*(Yes|No)\b", answer_block, flags=re.IGNORECASE)
    if not yn:
        return False
    return all(x.lower() == "yes" for x in yn)


def normalize_time_spans(text: str) -> str:
    """
    把:
        <t>0.0s-5.0s</t>
    变成:
        <t>0.0s</t> to <t>5.0s</t>
    """
    def repl(m):
        start_s = float(m.group(1))
        end_s = float(m.group(2))
        return f"<t>{start_s:.1f}s</t> to <t>{end_s:.1f}s</t>"

    return TIME_SPAN_PATTERN.sub(repl, text)


def find_recent_time_tag_start(thinking: str, noun: str, search_start: int = 0):
    """
    从 thinking[search_start:] 开始找 noun 的下一次出现，
    再向前找最近的原始时间段标签 <t>start-end</t>。
    返回:
        (start_sec, end_sec, noun_start_idx, noun_end_idx)
    """
    noun_match = re.search(re.escape(noun), thinking[search_start:])
    if not noun_match:
        return None

    abs_start = search_start + noun_match.start()
    abs_end = search_start + noun_match.end()

    prefix = thinking[:abs_start]
    time_matches = list(TIME_SPAN_PATTERN.finditer(prefix))
    if not time_matches:
        return None

    last_t = time_matches[-1]
    start_sec = float(last_t.group(1))
    end_sec = float(last_t.group(2))
    return start_sec, end_sec, abs_start, abs_end


def seconds_to_frame_index(start_sec: float, fps: float, num_frames: int) -> int:
    if num_frames <= 0:
        return 0
    frame_index = int(start_sec * fps)
    frame_index = max(0, min(frame_index, num_frames - 1))
    return frame_index


def replace_noun_at_position_with_box(text: str, noun: str, noun_start: int, noun_end: int, start_sec: float, box_xyxy):
    point_t = f"<t>{start_sec:.1f}s</t>"
    replacement = f"{noun} at {point_t}<box>{box_xyxy}</box>"
    return text[:noun_start] + replacement + text[noun_end:]


def collect_valid_noun_mentions(thinking: str, nouns: list):
    """
    为每个 noun 收集一次可处理的 mention：
    - noun 必须出现在 thinking 中
    - noun 前面必须存在最近的 <t>a-b</t>
    返回列表元素:
        {
            "noun": ...,
            "start_sec": ...,
            "end_sec": ...,
            "noun_start": ...,
            "noun_end": ...
        }
    """
    items = []
    used_spans = set()

    for noun in nouns:
        search_start = 0
        found_item = None

        while True:
            found = find_recent_time_tag_start(thinking, noun, search_start=search_start)
            if found is None:
                break

            start_sec, end_sec, noun_start, noun_end = found
            span_key = (noun, noun_start, noun_end)
            if span_key not in used_spans:
                found_item = {
                    "noun": noun,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "noun_start": noun_start,
                    "noun_end": noun_end,
                }
                used_spans.add(span_key)
                break

            search_start = noun_end

        if found_item is not None:
            items.append(found_item)

    # 按在原 thinking 中出现顺序处理，避免替换时顺序混乱
    items.sort(key=lambda x: x["noun_start"])
    return items


def process_one_sample(d, video_predictor):
    raw_sample = d["sample"]
    video_rel_path = raw_sample["videos"][0]
    video_path = os.path.join("data", video_rel_path)

    judgement = d.get("judgement")
    if judgement is None:
        judgement = raw_sample["conversations"][-1]["value"]

    thinking = d.get("thinking", "")
    nouns = d.get("nouns", [])

    # 先统一全文时间格式
    normalized_judgement = normalize_time_spans(judgement)
    normalized_thinking = normalize_time_spans(thinking)

    # 1) 如果 <answer> 中全是 Yes，直接跳过打标，但保留时间格式归一化
    if all_answers_yes(normalized_judgement):
        output = copy.deepcopy(raw_sample)
        output["conversations"][-1]["value"] = normalized_judgement
        return output

    # 2) 只保留在 thinking 中存在且前面有最近 <t>a-b</t> 的 noun
    valid_mentions = collect_valid_noun_mentions(thinking, nouns)

    if not valid_mentions:
        output = copy.deepcopy(raw_sample)
        output["conversations"][-1]["value"] = normalized_judgement
        return output

    session_id = None
    new_thinking = normalized_thinking

    try:
        response = video_predictor.handle_request(
            request={
                "type": "start_session",
                "resource_path": video_path,
            }
        )
        session_id = response.get("session_id")

        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

        if num_frames <= 0 or fps <= 0:
            output = copy.deepcopy(raw_sample)
            output["conversations"][-1]["value"] = normalized_judgement
            return output

        # 因为 normalized_thinking 长度和原 thinking 长度可能不同（时间标签被改写），
        # 所以不能直接用原始 noun_start/noun_end 在 normalized_thinking 上替换。
        # 这里改成在 current text 中逐个找 noun 的第一次未处理出现。
        for item in valid_mentions:
            noun = item["noun"]
            start_sec = item["start_sec"]

            frame_index = seconds_to_frame_index(start_sec, fps, num_frames)

            response = video_predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": frame_index,
                    "text": noun,
                }
            )

            output_data = response.get("outputs")
            if not output_data:
                continue

            boxes = output_data.get("out_boxes_xywh", None)
            if boxes is None:
                continue
            if isinstance(boxes, np.ndarray):
                if boxes.size == 0:
                    continue
            else:
                if len(boxes) == 0:
                    continue

            box = list(boxes[0])
            box = [int(b * 1000) for b in box]
            box = xywh_to_xyxy(box)

            noun_match = re.search(re.escape(noun), new_thinking)
            if noun_match is None:
                continue

            new_thinking = replace_noun_at_position_with_box(
                new_thinking,
                noun,
                noun_match.start(),
                noun_match.end(),
                start_sec,
                box,
            )

    finally:
        if session_id is not None:
            try:
                video_predictor.handle_request(
                    request={
                        "type": "close_session",
                        "session_id": session_id,
                    }
                )
            except Exception:
                pass

    output = copy.deepcopy(raw_sample)
    new_judgement = normalized_judgement.replace(normalized_thinking, new_thinking, 1)
    output["conversations"][-1]["value"] = new_judgement
    return output


def worker(rank, world_size, data, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    random.seed(1234 + rank)
    torch.manual_seed(1234 + rank)

    video_predictor = build_predictor(rank)
    shard = split_data(data, rank, world_size)

    jsonl_path = os.path.join(output_dir, f"train_spatial_temporal_rank{rank}.jsonl")
    done_path = os.path.join(output_dir, f"train_spatial_temporal_rank{rank}.done")
    err_path = os.path.join(output_dir, f"train_spatial_temporal_rank{rank}.errors.jsonl")

    done_set = load_done_set(done_path)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_fp, \
         open(done_path, "a", encoding="utf-8") as done_fp, \
         open(err_path, "a", encoding="utf-8") as err_fp:

        for d in tqdm(shard, desc=f"Rank {rank}", position=rank):
            try:
                sample_key = get_sample_key(d)
            except Exception as e:
                err_item = {
                    "key": None,
                    "video": d.get("video"),
                    "error": f"Failed to generate sample key: {str(e)}",
                }
                append_jsonl(err_fp, err_item)
                continue

            if sample_key in done_set:
                continue

            try:
                out = process_one_sample(d, video_predictor)
                append_jsonl(jsonl_fp, out)
                append_done(done_fp, sample_key)
                done_set.add(sample_key)
            except Exception as e:
                err_item = {
                    "key": sample_key,
                    "video": d.get("video"),
                    "error": str(e),
                }
                append_jsonl(err_fp, err_item)


def main():
    args = parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if args.merge_only:
        if local_rank == 0:
            merge_results(args.output_dir)
        return

    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    worker(local_rank, world_size, data, args.output_dir)


if __name__ == "__main__":
    main()