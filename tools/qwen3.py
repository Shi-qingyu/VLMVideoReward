import json
import copy
import os
import cv2
import random
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from sam3.model_builder import build_sam3_video_predictor

template = """This is a judgment of an AI-generated video. Please extract the nouns that refer to issues mentioned in the judgment. If there are multiple, separate them with commas. Judgement: {judgement}"""
template = """This is a judgment of an AI-generated video.

Your task is to extract ONLY the nouns that refer to issues mentioned in the judgment.

IMPORTANT RULES:
1. The nouns MUST be copied EXACTLY from the original judgment text.
2. DO NOT rephrase, summarize, or modify the words in any way.
3. Each extracted noun MUST be a direct substring of the judgment.
4. If multiple nouns exist, separate them with commas.

Judgment: {judgement}
"""

video_predictor = build_sam3_video_predictor()

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    # device_map="auto",   # 多卡/自动分配时建议加上
)
model.eval()

with open("data/train_fixed.json", "r", encoding="utf-8") as file:
    data = json.load(file)

outputs = []

with torch.inference_mode():
    for d in tqdm(data, desc="Processing data", unit="entry"):
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

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

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
            print(response)
            session_id = response.get("session_id")

            cap = cv2.VideoCapture(video_path)
            num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
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

                    output = response.get("outputs")
                    if not output:
                        continue

                    boxes = output.get("out_boxes_xywh", [])
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
        outputs.append(output)

        with open("data/train_region.json", "w", encoding="utf-8") as f:
            json.dump(outputs, f, ensure_ascii=False, indent=4)