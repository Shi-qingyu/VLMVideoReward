import os
import re
import json
import cv2
from pathlib import Path

INPUT_JSONL = "data/parallel_outputs/train_spatial_temporal_rank0.jsonl"
OUTPUT_DIR = "visualize"
VIDEO_PREFIX = "./data"
CONTEXT = 5  # 前后各取多少帧

PATTERN = re.compile(
    r"<timestamp>(\d+)</timestamp>\s*<region>\[([^\]]+)\]</region>",
    re.IGNORECASE,
)


def safe_stem(video_rel_path: str) -> str:
    p = Path(video_rel_path)
    return f"{p.parent.as_posix().replace('/', '__')}__{p.stem}"


def parse_pairs(gpt_text: str):
    pairs = []
    for m in PATTERN.finditer(gpt_text):
        frame_idx = int(m.group(1))
        nums = [int(float(x.strip())) for x in m.group(2).split(",")]
        if len(nums) != 4:
            continue
        pairs.append((frame_idx, nums))
    return pairs


def denorm_box_1000(box, img_w, img_h):
    x1, y1, x2, y2 = box
    x1 = int(round(x1 / 1000.0 * img_w))
    x2 = int(round(x2 / 1000.0 * img_w))
    y1 = int(round(y1 / 1000.0 * img_h))
    y2 = int(round(y2 / 1000.0 * img_h))
    return [x1, y1, x2, y2]


def clamp_box(box, w, h):
    x1, y1, x2, y2 = box
    x1 = max(0, min(w - 1, x1))
    x2 = max(0, min(w - 1, x2))
    y1 = max(0, min(h - 1, y1))
    y2 = max(0, min(h - 1, y2))
    return [x1, y1, x2, y2]


def draw_box(img, box, text):
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

    text_y = max(30, y1 - 10)
    cv2.putText(
        img,
        text,
        (x1, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return img


def save_annotated_clip(video_path, video_rel_path, pair_idx, frame_idx, box_1000):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 8.0

    start = max(0, frame_idx - CONTEXT)
    end = min(total_frames - 1, frame_idx + CONTEXT)

    if start > end:
        cap.release()
        return

    video_tag = safe_stem(video_rel_path)
    save_dir = os.path.join(OUTPUT_DIR, video_tag)
    os.makedirs(save_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first_frame = cap.read()
    if not ok or first_frame is None:
        print(f"[WARN] failed reading start frame {start} from {video_path}")
        cap.release()
        return

    h, w = first_frame.shape[:2]
    pixel_box = clamp_box(denorm_box_1000(box_1000, w, h), w, h)

    out_name = f"pair{pair_idx:02d}_ts{frame_idx:05d}.mp4"
    out_path = os.path.join(save_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # 写第一帧
    cur_idx = start
    label = f"ts={frame_idx} frame={cur_idx} box={pixel_box}"
    writer.write(draw_box(first_frame.copy(), pixel_box, label))

    # 写后续帧
    for cur_idx in range(start + 1, end + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[WARN] failed reading frame {cur_idx} from {video_path}")
            break
        label = f"ts={frame_idx} frame={cur_idx} box={pixel_box}"
        writer.write(draw_box(frame.copy(), pixel_box, label))

    writer.release()
    cap.release()


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = 0
    valid = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            total += 1
            try:
                item = json.loads(line)
            except Exception as e:
                print(f"[WARN] line {line_no}: invalid json: {e}")
                continue

            videos = item.get("videos", [])
            convs = item.get("conversations", [])
            if not videos or not convs:
                print(f"[WARN] line {line_no}: missing videos/conversations")
                continue

            video_rel_path = videos[0]
            video_path = os.path.join(VIDEO_PREFIX, video_rel_path)
            gpt_text = convs[-1].get("value", "")
            pairs = parse_pairs(gpt_text)

            if not pairs:
                print(f"[INFO] line {line_no}: no timestamp-region pairs")
                continue

            for pair_idx, (frame_idx, box_1000) in enumerate(pairs):
                save_annotated_clip(
                    video_path=video_path,
                    video_rel_path=video_rel_path,
                    pair_idx=pair_idx,
                    frame_idx=frame_idx,
                    box_1000=box_1000,
                )

            valid += 1
            print(f"[OK] line {line_no}: saved mp4 clips for {video_rel_path}, pairs={len(pairs)}")

    print(f"done. total={total}, valid_with_pairs={valid}")


if __name__ == "__main__":
    main()