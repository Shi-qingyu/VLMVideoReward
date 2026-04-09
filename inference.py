from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

# default: Load the model on the available device(s)
model = Qwen3VLForConditionalGeneration.from_pretrained(
    "output/qwen3vl-2b-baseline-1e-bs4-ga4-st-grpo/checkpoint-500", 
    dtype="auto", 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("output/qwen3vl-2b-baseline-1e-bs4-ga4-st-grpo/checkpoint-500")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "data/videos/eval_0/0.mp4",
            },
            {
                "type": "text", 
                "text": "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\nPlease watch the frames of a given video.\nEvaluate the video according to the following three dimensions.\n\n[Visual Quality]\nAssess the video in terms of:\nVideo Quality: whether the video is free from major visual defects, including blur, lack of detail, poor texture, lighting issues, color distortion, flickering, and overexposure.\n\n[Motion & Physical Consistency]\nAssess the video in terms of:\nSubject Movement: whether the subject's motion is natural, smooth, and physically realistic.\nPhysical Interaction: whether interactions among subjects and/or objects are physically plausible.\nCause-Effect: whether causal relationships are correctly depicted.\n\n[Prompt Alignment]\nTextual prompt: A confident young Latino man with a modern fade hairstyle stands alone in an urban outdoor setting. He wears a trendy oversized hoodie, tapered cargo pants, and high-top sneakers. He looks directly at the camera, maintaining steady eye contact as he speaks, his facial expressions showing subtle anticipation. The background features colorful street murals, graffiti art, and ambient city lights, highlighting the vibrant city atmosphere. The camera remains steady, capturing his bold street style and the dynamic urban environment..\nAssess whether the video is well-aligned with the textual prompt in terms of:\nSubject Existence: whether the subject described in the prompt appears and is accurate.\nObject Existence: whether the object described in the prompt appears and is accurate.\nSubject-Object Interaction: whether the interaction described in the prompt is correctly represented.\n\nProvide your reasoning, then output \"Yes\" or \"No\"."
            },
        ],
    }
]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    fps=2
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=1024)
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text[0])
