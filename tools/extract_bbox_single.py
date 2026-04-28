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
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


TIME_SPAN_PATTERN = re.compile(
    r"<t>\s*([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\s*</t>",
    flags=re.IGNORECASE,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, default="data/train_nouns.json")
    parser.add_argument("--output_dir", type=str, default="data/parallel_outputs")
    parser.add_argument(
        "--merge_only",
        action="store_true",
        default=False,
        help="Only merge results without processing",
    )

    # 新增：SAM3 image 推理相关参数
    parser.add_argument(
        "--min_score",
        type=float,
        default=0.0,
        help="Minimum score for accepting SAM3 image prediction. Default keeps all predictions.",
    )
    parser.add_argument(
        "--prefer_mask_box",
        action="store_true",
        default=True,
        help="Prefer bbox computed from mask when masks are available.",
    )
    parser.add_argument(
        "--no_prefer_mask_box",
        dest="prefer_mask_box",
        action="store_false",
        help="Use SAM3 returned boxes first instead of computing boxes from masks.",
    )

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

    print(f"Merged {len(merged)} samples to {out_path}")


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


def build_image_processor(rank):
    """
    构建 SAM3 image model + processor。

    官方基础用法：
        model = build_sam3_image_model()
        processor = Sam3Processor(model)

    多卡场景下，这里先 set_device(rank)。
    如果你的 SAM3 build 函数内部已经自动放到当前 CUDA 设备，
    这段就足够；如果没有，下面会尝试 model.to(device)。
    """
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f"cuda:{rank}")
    else:
        device = torch.device("cpu")

    model = build_sam3_image_model()

    try:
        model = model.to(device)
    except Exception:
        # 某些封装模型可能没有标准 .to()，忽略即可
        pass

    try:
        model.eval()
    except Exception:
        pass

    processor = Sam3Processor(model)
    return processor


def read_video_meta(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return 0, 0.0

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))

    cap.release()
    return num_frames, fps


def read_frame_as_pil(video_path, frame_index):
    """
    从视频中抽取指定帧，返回 PIL RGB Image。
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        cap.release()
        return None

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame_bgr = cap.read()
    cap.release()

    if not ok or frame_bgr is None:
        return None

    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    return image


def xywh_to_xyxy(box):
    x_min, y_min, w, h = box
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def clamp_box_1000(box):
    return [int(max(0, min(1000, round(float(x))))) for x in box]


def box_xyxy_to_1000(box, image_w, image_h):
    """
    将 SAM3 返回的 box 转成 0~1000 的 xyxy。

    兼容两种常见情况：
    1. box 已经是 0~1 归一化坐标；
    2. box 是原图像素坐标。
    """
    box = np.asarray(box, dtype=np.float32).reshape(-1).tolist()

    if len(box) < 4:
        return None

    x1, y1, x2, y2 = box[:4]

    # 如果看起来像 0~1 归一化坐标
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.5:
        out = [
            x1 * 1000.0,
            y1 * 1000.0,
            x2 * 1000.0,
            y2 * 1000.0,
        ]
    else:
        # 默认认为是像素 xyxy
        out = [
            x1 / image_w * 1000.0,
            y1 / image_h * 1000.0,
            x2 / image_w * 1000.0,
            y2 / image_h * 1000.0,
        ]

    return clamp_box_1000(out)


def mask_to_box_xyxy_1000(mask, image_w=None, image_h=None):
    """
    从 mask 计算 0~1000 的 xyxy bbox。

    mask 支持：
    - torch.Tensor
    - np.ndarray
    - shape: H x W
    - shape: 1 x H x W
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().float().cpu().numpy()

    mask = np.asarray(mask)

    # squeeze 到 H x W
    while mask.ndim > 2:
        mask = np.squeeze(mask, axis=0)

    if mask.ndim != 2:
        return None

    mask_bin = mask > 0

    ys, xs = np.where(mask_bin)

    if len(xs) == 0 or len(ys) == 0:
        return None

    h, w = mask.shape[:2]

    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()

    out = [
        x1 / w * 1000.0,
        y1 / h * 1000.0,
        x2 / w * 1000.0,
        y2 / h * 1000.0,
    ]

    return clamp_box_1000(out)


def tensor_or_array_len(x):
    if x is None:
        return 0
    if isinstance(x, torch.Tensor):
        if x.ndim == 0:
            return 1
        return x.shape[0]
    if isinstance(x, np.ndarray):
        if x.ndim == 0:
            return 1
        return x.shape[0]
    return len(x)


def get_item(x, idx):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x[idx]
    if isinstance(x, np.ndarray):
        return x[idx]
    return x[idx]


def score_to_float(score):
    if score is None:
        return 0.0

    if isinstance(score, torch.Tensor):
        return float(score.detach().cpu().reshape(-1)[0])

    if isinstance(score, np.ndarray):
        return float(score.reshape(-1)[0])

    return float(score)


def sam3_image_text_prompt_to_box(
    processor,
    image,
    text_prompt,
    min_score=0.0,
    prefer_mask_box=True,
):
    """
    对单张图片用 SAM3 text prompt 分割/检测，返回一个 0~1000 xyxy box。

    返回:
        [x1, y1, x2, y2] 或 None
    """
    image_w, image_h = image.size

    with torch.inference_mode():
        inference_state = processor.set_image(image)
        output = processor.set_text_prompt(
            state=inference_state,
            prompt=text_prompt,
        )

    masks = output.get("masks", None)
    boxes = output.get("boxes", None)
    scores = output.get("scores", None)

    n_masks = tensor_or_array_len(masks)
    n_boxes = tensor_or_array_len(boxes)
    n_scores = tensor_or_array_len(scores)

    n = max(n_masks, n_boxes, n_scores)

    if n <= 0:
        return None

    # 选 score 最高的实例；如果没有 scores，就选第一个
    best_idx = 0
    best_score = None

    if n_scores > 0:
        best_score = -1e9
        for i in range(n_scores):
            s = score_to_float(get_item(scores, i))
            if s > best_score:
                best_score = s
                best_idx = i

        if best_score < min_score:
            return None

    # 优先从 mask 算 box，一般更贴合分割结果
    if prefer_mask_box and n_masks > best_idx:
        box = mask_to_box_xyxy_1000(
            get_item(masks, best_idx),
            image_w=image_w,
            image_h=image_h,
        )
        if box is not None:
            return box

    # 其次使用 SAM3 返回的 boxes
    if n_boxes > best_idx:
        box = box_xyxy_to_1000(
            get_item(boxes, best_idx),
            image_w=image_w,
            image_h=image_h,
        )
        if box is not None:
            return box

    # 如果上面 boxes 失败，再 fallback 到 mask
    if n_masks > best_idx:
        box = mask_to_box_xyxy_1000(
            get_item(masks, best_idx),
            image_w=image_w,
            image_h=image_h,
        )
        if box is not None:
            return box

    return None


def extract_answer_block(text):
    m = re.search(
        r"<answer>\s*(.*?)\s*</answer>",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    return m.group(1) if m else ""


def all_answers_yes(text):
    answer_block = extract_answer_block(text)

    if not answer_block:
        return False

    yn = re.findall(r":\s*(Yes|No)\b", answer_block, flags=re.IGNORECASE)

    if not yn:
        return False

    return all(x.lower() == "yes" for x in yn)


def normalize_time_spans(text):
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


def find_recent_time_tag_start(thinking, noun, search_start=0):
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


def seconds_to_frame_index(start_sec, fps, num_frames):
    if num_frames <= 0:
        return 0

    frame_index = int(start_sec * fps)
    frame_index = max(0, min(frame_index, num_frames - 1))

    return frame_index


def replace_noun_at_position_with_box(
    text,
    noun,
    noun_start,
    noun_end,
    start_sec,
    box_xyxy,
):
    point_t = f"<t>{start_sec:.1f}s</t>"
    replacement = f"{noun} at {point_t}<box>{box_xyxy}</box>"

    return text[:noun_start] + replacement + text[noun_end:]


def collect_valid_noun_mentions(thinking, nouns):
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
            found = find_recent_time_tag_start(
                thinking,
                noun,
                search_start=search_start,
            )

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

    items.sort(key=lambda x: x["noun_start"])
    return items


def find_next_unboxed_noun(text, noun, search_start=0):
    """
    在当前 new_thinking 中找下一个还没有被替换成 bbox 的 noun。

    避免重复 noun 时总是替换第一个 noun。
    简单策略：
    - 从 search_start 后找 noun
    - 如果 noun 后面紧跟 " at <t>...<box>"，认为已经处理过，继续找
    """
    pos = search_start

    while True:
        m = re.search(re.escape(noun), text[pos:])

        if m is None:
            return None

        abs_start = pos + m.start()
        abs_end = pos + m.end()

        tail = text[abs_end:abs_end + 80]

        if re.match(r"\s+at\s+<t>[0-9]+(?:\.[0-9]+)?s</t>\s*<box>", tail):
            pos = abs_end
            continue

        return abs_start, abs_end


def process_one_sample(
    d,
    image_processor,
    min_score=0.0,
    prefer_mask_box=True,
):
    raw_sample = d["sample"]
    video_rel_path = raw_sample["videos"][0]
    video_path = os.path.join("data", video_rel_path)

    judgement = d.get("judgement")
    if judgement is None:
        judgement = raw_sample["conversations"][-1]["value"]

    thinking = d.get("thinking", "")
    nouns = d.get("nouns", [])

    normalized_judgement = normalize_time_spans(judgement)
    normalized_thinking = normalize_time_spans(thinking)

    # 1) 如果 <answer> 中全是 Yes，直接跳过打标，但保留时间格式归一化
    if all_answers_yes(normalized_judgement):
        output = copy.deepcopy(raw_sample)
        output["conversations"][-1]["value"] = normalized_judgement
        return output

    # 2) 只保留在原 thinking 中存在且前面有最近 <t>a-b</t> 的 noun
    valid_mentions = collect_valid_noun_mentions(thinking, nouns)

    if not valid_mentions:
        output = copy.deepcopy(raw_sample)
        output["conversations"][-1]["value"] = normalized_judgement
        return output

    num_frames, fps = read_video_meta(video_path)

    if num_frames <= 0 or fps <= 0:
        output = copy.deepcopy(raw_sample)
        output["conversations"][-1]["value"] = normalized_judgement
        return output

    new_thinking = normalized_thinking

    # 同一个视频同一帧可能有多个 noun，缓存抽帧结果，避免重复 decode
    frame_cache = {}

    # 为了避免 duplicate noun 替换错位置，维护一个搜索游标
    search_cursor = 0

    for item in valid_mentions:
        noun = item["noun"]
        start_sec = item["start_sec"]

        frame_index = seconds_to_frame_index(
            start_sec=start_sec,
            fps=fps,
            num_frames=num_frames,
        )

        cache_key = frame_index

        if cache_key not in frame_cache:
            frame_cache[cache_key] = read_frame_as_pil(video_path, frame_index)

        image = frame_cache[cache_key]

        if image is None:
            continue

        try:
            box = sam3_image_text_prompt_to_box(
                processor=image_processor,
                image=image,
                text_prompt=noun,
                min_score=min_score,
                prefer_mask_box=prefer_mask_box,
            )
        except Exception:
            # 单个 noun 失败不要中断整个样本
            continue

        if box is None:
            continue

        found = find_next_unboxed_noun(
            text=new_thinking,
            noun=noun,
            search_start=search_cursor,
        )

        # 如果从 cursor 后没找到，就从全文 fallback 找一次
        if found is None:
            found = find_next_unboxed_noun(
                text=new_thinking,
                noun=noun,
                search_start=0,
            )

        if found is None:
            continue

        noun_start, noun_end = found

        new_thinking = replace_noun_at_position_with_box(
            text=new_thinking,
            noun=noun,
            noun_start=noun_start,
            noun_end=noun_end,
            start_sec=start_sec,
            box_xyxy=box,
        )

        search_cursor = noun_start + len(
            f"{noun} at <t>{start_sec:.1f}s</t><box>{box}</box>"
        )

    output = copy.deepcopy(raw_sample)

    # 优先替换 judgement 中的 thinking block
    if normalized_thinking in normalized_judgement:
        new_judgement = normalized_judgement.replace(
            normalized_thinking,
            new_thinking,
            1,
        )
    else:
        # 如果因为数据不一致导致 replace 失败，保守地只返回 normalized_judgement
        # 也可以改成直接把 new_thinking append 到 judgement，视你的数据格式而定
        new_judgement = normalized_judgement

    output["conversations"][-1]["value"] = new_judgement
    return output


def worker(
    rank,
    world_size,
    data,
    output_dir,
    min_score=0.0,
    prefer_mask_box=True,
):
    os.makedirs(output_dir, exist_ok=True)

    random.seed(1234 + rank)
    torch.manual_seed(1234 + rank)

    image_processor = build_image_processor(rank)
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
                out = process_one_sample(
                    d=d,
                    image_processor=image_processor,
                    min_score=min_score,
                    prefer_mask_box=prefer_mask_box,
                )

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

    worker(
        rank=local_rank,
        world_size=world_size,
        data=data,
        output_dir=args.output_dir,
        min_score=args.min_score,
        prefer_mask_box=args.prefer_mask_box,
    )


if __name__ == "__main__":
    main()