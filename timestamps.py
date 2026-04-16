import json
import os
import torch
import copy
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

OPTIMIZED_TEMPLATE = "Identify the exact timestamps when the {nouns} exhibit anomalous behavior in the video. Provide the results as normalized values between 0.0 and 1.0. Return ONLY the numerical values separated by commas, with no additional text."

def load_data(json_file):
    if not os.path.exists(json_file):
        raise FileNotFoundError(f"Data file not found: {json_file}")
    with open(json_file, "r") as f:
        return json.load(f)

def load_model(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, 
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    return model, processor

@torch.inference_mode()
def run_inference(model, processor, video_path, nouns):
    prompt = OPTIMIZED_TEMPLATE.format(nouns=nouns)
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": video_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[text],
        videos=[video_path],
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    # 生成
    generated_ids = model.generate(
        **inputs, 
        max_new_tokens=128,
        do_sample=False
    )
    
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    
    try:
        # timestamps = [float(ts.strip()) for ts in output_text.split(",") if ts.strip()]
        return output_text
    except Exception as e:
        print(f"error video: {video_path}, output: {output_text}, error: {e}")
        return []

def main(data_file, model_path, output_file):
    examples = load_data(data_file)
    model, processor = load_model(model_path)
    
    final_results = []

    for example in tqdm(examples, desc="Annotating Videos"):
        res_entry = copy.deepcopy(example)
        
        
        nouns = example["nouns_raw"]
        video_path = os.path.join("data", example["video"])
        
        timestamps: list = run_inference(model, processor, video_path, nouns)
        
        res_entry["timestamps"] = timestamps
        final_results.append(res_entry)

        if len(final_results) % 1 == 0:
            with open(output_file, "w") as f:
                json.dump(final_results, f, indent=4)

    with open(output_file, "w") as f:
        json.dump(final_results, f, indent=4)
    print(f"Done! Results saved to {output_file}")

# 如果需要作为脚本运行
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str, default="data/train_nouns.json")
    parser.add_argument("--model_path", type=str, default="Qwen/Qwen3-VL-30B-A3B-Instruct")
    parser.add_argument("--output_file", type=str, default="annotated_results.json")
    args = parser.parse_args()
    
    main(args.data_file, args.model_path, args.output_file)