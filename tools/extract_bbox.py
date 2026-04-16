import os
import json
import copy
import random
import hashlib
import argparse
import cv2
import torch
import glob
import os
import json

from tqdm import tqdm
from sam3.model_builder import build_sam3_video_predictor


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


def process_one_sample(d, video_predictor):
    raw_sample = d["sample"]
    video_path = os.path.join("data", raw_sample["videos"][0])

    judgement = d["judgement"]
    thinking = d["thinking"]
    texts = d.get("nouns", [])
    timestamps = d.get("timestamps", [])

    if len(timestamps) > len(texts):
        timestamps = timestamps[:len(texts)]
    elif len(timestamps) < len(texts):
        timestamps = timestamps + [0.0] * (len(texts) - len(timestamps))

    new_thinking = thinking
    session_id = None

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
        cap.release()

        if num_frames <= 0:
            output = copy.deepcopy(raw_sample)
            return output

        if session_id:
            for frame_index, text_item in zip(timestamps, texts):
                frame_index = int(frame_index * num_frames)
                frame_index = frame_index if frame_index < num_frames else num_frames - 1
                response = video_predictor.handle_request(
                    request={
                        "type": "add_prompt",
                        "session_id": session_id,
                        "frame_index": frame_index,
                        "text": text_item,
                    }
                )

                output_data = response.get("outputs")
                if not output_data:
                    continue

                boxes = output_data.get("out_boxes_xywh", [])
                if len(boxes) == 0:
                    continue

                box = list(boxes[0])
                box = [int(b * 1000) for b in box]
                box = xywh_to_xyxy(box)
                new_text = f"At <timestamp>{frame_index}</timestamp><region>{box}</region> the {text_item}"
                if new_text not in new_thinking:
                    new_thinking = new_thinking.replace(text_item, new_text, 1)

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
    new_judgement = judgement.replace(thinking, new_thinking, 1)
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
    else:
        with open(args.input_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        worker(local_rank, world_size, data, args.output_dir)


if __name__ == "__main__":
    main()