import json
import os
import copy
import re

from tqdm import tqdm


OUTPUT_DIR = "data/parallel_outputs"


def merge_results(world_size):
    merged = []
    for rank in range(world_size):
        path = os.path.join(OUTPUT_DIR, f"train_region_rank{rank}.jsonl")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        merged.append(json.loads(line))

    print(f"Merged {len(merged)} entries from {world_size} files.")
    with open(os.path.join("data", "train_region.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)


def extract_tag_content(text: str, tag: str) -> str:
    """
    Extract content inside <tag>...</tag>. Return empty string if not found.
    """
    pattern = rf"<{tag}>\s*(.*?)\s*</{tag}>"
    matches = re.findall(pattern, text, re.S | re.I)
    return [match.strip() for match in matches] if matches else []


def parse_box(box_str: str):
    """
    Parse a box string like "[x1,y1,x2,y2]" into a list of integers [x1, y1, x2, y2].
    """
    return json.loads(box_str)


def xywh_to_xyxy(box: list):
    x_min, y_min, w, h = box
    x_max = x_min + w
    y_max = y_min + h
    return [x_min, y_min, x_max, y_max]


def format_bboxes(file_path):
    with open(file_path, "r") as file:
        examples = json.load(file)
    
    new_examples = []
    cnt = 0
    for example in tqdm(examples):
        new_example = copy.deepcopy(example)
        ground_truth = example["conversations"][-1]["value"]
        new_ground_truth = ground_truth
        bboxes = extract_tag_content(ground_truth, "region")
        if len(bboxes) == 0:
            new_examples.append(new_example)
        else:
            for box in bboxes:
                box_list = parse_box(box)
                box_list = [int(b * 1000) for b in box_list]
                box_list = xywh_to_xyxy(box_list)
                box_list = str(box_list)
                new_ground_truth = new_ground_truth.replace(box, box_list)
            new_example["conversations"][-1]["value"] = new_ground_truth
            new_examples.append(new_example)
        
    
    with open("data/train_region_new.json", "w") as file:
        json.dump(new_examples, file, indent=4)
        
    
if __name__ == "__main__":
    world_size = 8  # Adjust this based on your setup
    merge_results(world_size)
    
    format_bboxes("data/train_region.json")