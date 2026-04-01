import argparse
import os
import json
import cv2
from typing import List, Optional, Tuple
from tqdm import tqdm
from qwenvl.train.utils import extract_tag_content, parse_box

ROOT = "data"
SAVE_ROOT = "data/eval_w_boxes"
os.makedirs(SAVE_ROOT, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Draw bounding boxes on video frames.")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON file containing video paths and bounding boxes.")
    return parser.parse_args()

def draw_bboxes_on_video(
    video_path: str,
    bboxes: List[List[int]],
    output_path: Optional[str] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> str:
    """
    Draw all bounding boxes on every frame of a video and save the result.

    Args:
        video_path: Path to the input video.
        bboxes: List of boxes in [x1, y1, x2, y2] format.
                Coordinates are normalized to the range [0, 1000].
        output_path: Path to the output video. If None, auto-generate one.
        color: Bounding box color in BGR format.
        thickness: Line thickness for the boxes.

    Returns:
        Path to the output video.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not isinstance(bboxes, list):
        raise TypeError("bboxes must be a list of [x1, y1, x2, y2]")

    for i, box in enumerate(bboxes):
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            raise ValueError(f"bboxes[{i}] must be [x1, y1, x2, y2]")
        if not all(isinstance(v, int) for v in box):
            raise ValueError(f"bboxes[{i}] must contain integers only")
        if not all(0 <= v <= 1000 for v in box):
            raise ValueError(f"bboxes[{i}] values must be in [0, 1000]")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Invalid video resolution")

    if output_path is None:
        base, _ = os.path.splitext(video_path)
        output_path = f"{base}_boxed.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Could not create output video: {output_path}")

    scaled_bboxes = []
    for box in bboxes:
        x1, y1, x2, y2 = box

        px1 = int(round(x1 / 1000.0 * width))
        py1 = int(round(y1 / 1000.0 * height))
        px2 = int(round(x2 / 1000.0 * width))
        py2 = int(round(y2 / 1000.0 * height))

        px1 = max(0, min(px1, width - 1))
        py1 = max(0, min(py1, height - 1))
        px2 = max(0, min(px2, width - 1))
        py2 = max(0, min(py2, height - 1))

        if px1 > px2:
            px1, px2 = px2, px1
        if py1 > py2:
            py1, py2 = py2, py1

        scaled_bboxes.append((px1, py1, px2, py2))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        for px1, py1, px2, py2 in scaled_bboxes:
            cv2.rectangle(frame, (px1, py1), (px2, py2), color, thickness)

        writer.write(frame)

    cap.release()
    writer.release()
    return output_path


def main():
    args = parse_args()

    with open("data/eval_filtered.json", "r") as f:
        paths = json.load(f)

    with open(args.input_file, 'r') as f:
        data = json.load(f)

    for path, item in tqdm(zip(paths, data), total=len(data)):
        video_path = os.path.join(ROOT, path["videos"][0])
        box_str = extract_tag_content(item['answer'], "region")

        if box_str[0] != "":
            bboxes = []
            for box in box_str:
                bboxes.append(parse_box(box))
            output_path = draw_bboxes_on_video(video_path, bboxes, output_path=os.path.join(SAVE_ROOT, os.path.basename(video_path)))
            print(f"Saved boxed video to: {output_path}")


if __name__ == "__main__":
    main()