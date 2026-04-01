import os
import json
import copy
import random
import hashlib
import cv2
import torch
import torch.multiprocessing as mp
import torch.distributed as dist

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sam3.model_builder import build_sam3_video_predictor

template = """This is a judgment of an AI-generated video.

Your task is to extract ONLY the nouns that refer to issues mentioned in the judgment.

IMPORTANT RULES:
1. The nouns MUST be copied EXACTLY from the original judgment text.
2. DO NOT rephrase, summarize, or modify the words in any way.
3. Each extracted noun MUST be a direct substring of the judgment.
4. If multiple nouns exist, separate them with commas.

Judgment: {judgement}
"""

MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507"
INPUT_JSON = "data/train_fixed.json"
OUTPUT_DIR = "data/parallel_outputs"


def split_data(data, rank, world_size):
    return data[rank::world_size]


def build_models(rank):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16,
        device_map={"": device},
        low_cpu_mem_usage=True,
    )
    model.eval()

    video_predictor = build_sam3_video_predictor(
        gpus_to_use=[rank]
    )
    return tokenizer, model, video_predictor, device


def get_sample_key(d):
    if "id" in d:
        return str(d["id"])

    raw = json.dumps(
        {
            "video": d["videos"][0] if d.get("videos") else "",
            "judgement": d["conversations"][-1]["value"] if d.get("conversations") else "",
        },
        ensure_ascii=False,
        sort_keys=True,
    )
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


def process_one_sample(d, tokenizer, model, video_predictor, device):
    video_path = os.path.join("data", d["videos"][0])
    judgement = d["conversations"][-1]["value"]

    start_pos = judgement.find("<think>")
    end_pos = judgement.find("</think>")

    if start_pos != -1 and end_pos != -1 and end_pos > start_pos:
        thinking = judgement[start_pos + len("<think>"): end_pos]
    else:
        thinking = judgement

    new_thinking = thinking

    prompt = template.format(judgement=thinking)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    with torch.inference_mode():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512,
            do_sample=False,
        )

    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):]
    content = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

    del model_inputs, generated_ids, output_ids
    torch.cuda.empty_cache()

    texts = [x.strip() for x in content.split(",") if x.strip()]
    texts = list(dict.fromkeys(texts))

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
            output = copy.deepcopy(d)
            return output

        frame_index = random.randint(0, num_frames - 1)

        if session_id:
            for text_item in texts:
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
                new_text = f"{text_item} at <region>{box}</region>"
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

    output = copy.deepcopy(d)
    new_judgement = judgement.replace(thinking, new_thinking, 1)
    output["conversations"][-1]["value"] = new_judgement
    return output


def worker(rank, world_size, data):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    random.seed(1234 + rank)
    torch.manual_seed(1234 + rank)

    tokenizer, model, video_predictor, device = build_models(rank)

    shard = split_data(data, rank, world_size)

    jsonl_path = os.path.join(OUTPUT_DIR, f"train_region_rank{rank}.jsonl")
    done_path = os.path.join(OUTPUT_DIR, f"train_region_rank{rank}.done")
    err_path = os.path.join(OUTPUT_DIR, f"train_region_rank{rank}.errors.jsonl")

    done_set = load_done_set(done_path)

    with open(jsonl_path, "a", encoding="utf-8") as jsonl_fp, \
         open(done_path, "a", encoding="utf-8") as done_fp, \
         open(err_path, "a", encoding="utf-8") as err_fp:

        with torch.inference_mode():
            for d in tqdm(shard, desc=f"Rank {rank}", position=rank):
                sample_key = get_sample_key(d)

                if sample_key in done_set:
                    continue

                try:
                    out = process_one_sample(d, tokenizer, model, video_predictor, device)
                    append_jsonl(jsonl_fp, out)
                    append_done(done_fp, sample_key)
                    done_set.add(sample_key)
                except Exception as e:
                    err_item = {
                        "key": sample_key,
                        "video": d["videos"][0] if d.get("videos") else None,
                        "error": str(e),
                    }
                    append_jsonl(err_fp, err_item)


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

    with open(os.path.join(OUTPUT_DIR, "train_region.json"), "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    worker(local_rank, world_size, data)

    # if local_rank == 0:
    #     merge_results(world_size)