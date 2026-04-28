import os
import re
import json
import cv2
from pathlib import Path

INPUT_JSON = "data/train_t_polished_v3.json"
OUTPUT_DIR = "visualize"
VIDEO_PREFIX = "./data"
CONTEXT = 15  # 目标帧前后各取多少帧

TIME_PATTERN = re.compile(
    r"<t>\s*([0-9]+(?:\.[0-9]+)?)s\s*-\s*([0-9]+(?:\.[0-9]+)?)s\s*</t>",
    re.IGNORECASE,
)


def safe_stem(video_rel_path: str) -> str:
    p = Path(video_rel_path)
    parent = p.parent.as_posix().replace("/", "__")
    if parent:
        return f"{parent}__{p.stem}"
    return p.stem


def extract_first_sentence_after(text: str, start_pos: int):
    """
    提取 <t>...</t> 后面的第一句话。
    例如:
        <t>1.8s-5.0s</t> Flickering issue
    返回:
        Flickering issue
    """
    tail = text[start_pos:].strip()
    if not tail:
        return ""

    # 遇到下一个标签就截断，避免串到后面的 </think> / <answer> / 下一个 <t>
    next_tag = tail.find("<")
    if next_tag != -1:
        tail = tail[:next_tag].strip()

    # 优先取到第一个句号、问号、感叹号
    m = re.search(r"(.+?[。！？.!?])(?:\s|$)", tail)
    if m:
        return m.group(1).strip()

    # 没有句末符号就取第一行
    lines = tail.splitlines()
    if lines:
        return lines[0].strip()

    return tail.strip()


def parse_time_spans(gpt_text: str, fps: float, total_frames: int):
    """
    解析:
        <t>0.0s-5.0s</t> 后面的一句话

    返回:
        [(frame_idx, start_sec, end_sec, desc), ...]

    默认抽 start_sec 对应帧。
    如果想抽时间段中间帧，把:
        target_sec = start_sec
    改成:
        target_sec = (start_sec + end_sec) / 2
    """
    pairs = []

    for m in TIME_PATTERN.finditer(gpt_text):
        start_sec = float(m.group(1))
        end_sec = float(m.group(2))

        target_sec = start_sec
        frame_idx = int(target_sec * fps)

        if total_frames > 0:
            frame_idx = max(0, min(frame_idx, total_frames - 1))

        desc = extract_first_sentence_after(gpt_text, m.end())
        pairs.append((frame_idx, start_sec, end_sec, desc))

    return pairs


def wrap_text_by_pixel(text, font, font_scale, thickness, max_width):
    """
    按像素宽度换行，避免文字超出画面。
    """
    if not text:
        return []

    words = text.split()
    if not words:
        return [text]

    lines = []
    cur = ""

    for word in words:
        test = word if not cur else cur + " " + word
        width, _ = cv2.getTextSize(test, font, font_scale, thickness)[0]

        if width <= max_width:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word

    if cur:
        lines.append(cur)

    return lines


def draw_label(img, header_text, desc_text=""):
    """
    在视频帧上贴:
    1. 时间和帧号
    2. <t> 后面的一句话

    带黑色背景，避免文字看不清。
    """
    h, w = img.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX
    header_scale = 0.75
    desc_scale = 0.68
    thickness = 2
    line_gap = 28

    x = 16
    y = 32
    max_text_width = max(100, w - 2 * x)

    lines = [header_text]
    desc_lines = wrap_text_by_pixel(
        desc_text,
        font=font,
        font_scale=desc_scale,
        thickness=thickness,
        max_width=max_text_width,
    )
    lines.extend(desc_lines)

    # 计算背景高度
    box_h = 18 + len(lines) * line_gap
    box_w = w
    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (box_w, min(h, box_h)),
        (0, 0, 0),
        -1,
    )

    # 半透明黑底
    alpha = 0.55
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # 画 header
    cv2.putText(
        img,
        header_text,
        (x, y),
        font,
        header_scale,
        (0, 255, 0),
        thickness,
        cv2.LINE_AA,
    )

    # 画描述
    for i, line in enumerate(desc_lines):
        yy = y + (i + 1) * line_gap
        cv2.putText(
            img,
            line,
            (x, yy),
            font,
            desc_scale,
            (0, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return img


def save_clip(video_path, video_rel_path, pair_idx, frame_idx, start_sec, end_sec, desc):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[WARN] cannot open video: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or fps <= 0:
        fps = 8.0

    start = max(0, frame_idx - CONTEXT)
    end = min(total_frames - 1, frame_idx + CONTEXT)

    if start > end:
        cap.release()
        return False

    video_tag = safe_stem(video_rel_path)
    save_dir = os.path.join(OUTPUT_DIR, video_tag)
    os.makedirs(save_dir, exist_ok=True)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ok, first_frame = cap.read()

    if not ok or first_frame is None:
        print(f"[WARN] failed reading start frame {start} from {video_path}")
        cap.release()
        return False

    h, w = first_frame.shape[:2]

    out_name = f"pair{pair_idx:02d}_t{start_sec:.1f}s-{end_sec:.1f}s_f{frame_idx:05d}.mp4"
    out_name = out_name.replace(":", "_")
    out_path = os.path.join(save_dir, out_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    cur_idx = start
    header = f"t={start_sec:.1f}-{end_sec:.1f}s target={frame_idx} frame={cur_idx}"
    writer.write(draw_label(first_frame.copy(), header, desc_text=desc))

    for cur_idx in range(start + 1, end + 1):
        ok, frame = cap.read()

        if not ok or frame is None:
            print(f"[WARN] failed reading frame {cur_idx} from {video_path}")
            break

        header = f"t={start_sec:.1f}-{end_sec:.1f}s target={frame_idx} frame={cur_idx}"
        writer.write(draw_label(frame.copy(), header, desc_text=desc))

    writer.release()
    cap.release()

    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected input json to be a list, got {type(data)}")

    total = 0
    valid = 0
    saved = 0

    for item_idx, item in enumerate(data):
        total += 1
        item_no = item_idx + 1

        if not isinstance(item, dict):
            print(f"[WARN] item {item_no}: expected dict, got {type(item)}")
            continue

        videos = item.get("videos", [])
        convs = item.get("conversations", [])

        if not videos or not convs:
            print(f"[WARN] item {item_no}: missing videos/conversations")
            continue

        video_rel_path = videos[0]
        video_path = os.path.join(VIDEO_PREFIX, video_rel_path)
        gpt_text = convs[-1].get("value", "")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[WARN] item {item_no}: cannot open video: {video_path}")
            continue

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()

        if not fps or fps <= 0:
            fps = 8.0

        pairs = parse_time_spans(
            gpt_text=gpt_text,
            fps=fps,
            total_frames=total_frames,
        )

        if not pairs:
            print(f"[INFO] item {item_no}: no time spans")
            continue

        valid += 1

        for pair_idx, (frame_idx, start_sec, end_sec, desc) in enumerate(pairs):
            ok = save_clip(
                video_path=video_path,
                video_rel_path=video_rel_path,
                pair_idx=pair_idx,
                frame_idx=frame_idx,
                start_sec=start_sec,
                end_sec=end_sec,
                desc=desc,
            )

            if ok:
                saved += 1

        print(
            f"[OK] item {item_no}: video={video_rel_path}, "
            f"time_spans={len(pairs)}, fps={fps:.3f}"
        )

    print(
        f"done. total={total}, "
        f"valid_with_time_spans={valid}, "
        f"saved_mp4={saved}"
    )


if __name__ == "__main__":
    main()