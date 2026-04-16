from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "output/qwen3vl-2b-baseline-1e-bs4-ga4-region/checkpoint-697", 
    dtype="auto", 
    device_map="auto"
)

processor = AutoProcessor.from_pretrained("output/qwen3vl-2b-baseline-1e-bs4-ga4-region/checkpoint-697")

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
                "text": "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\nPlease watch the frames of a given video.\nEvaluate the video according to the following three dimensions.\n\n[Visual Quality]\nAssess the video in terms of:\nVideo Quality: whether the video is free from major visual defects, including blur, lack of detail, poor texture, lighting issues, color distortion, flickering, and overexposure.\n\n[Motion & Physical Consistency]\nAssess the video in terms of:\nSubject Movement: whether the subject's motion is natural, smooth, and physically realistic.\nPhysical Interaction: whether interactions among subjects and/or objects are physically plausible.\nCause-Effect: whether causal relationships are correctly depicted.\n\n[Prompt Alignment]\nTextual prompt: A Black man in a short-sleeve shirt stands at a kitchen stove. He holds a box of dry pasta in one hand and pours the pasta into a pot of boiling water. Using a wooden spoon, he gently presses the pasta down to ensure it is fully submerged. The background shows kitchen counters and utensils. The camera remains steady, focusing on the man’s hands and the pot..\nAssess whether the video is well-aligned with the textual prompt in terms of:\nSubject Existence: whether the subject described in the prompt appears and is accurate.\nObject Existence: whether the object described in the prompt appears and is accurate.\nSubject-Object Interaction: whether the interaction described in the prompt is correctly represented.\n\nProvide your reasoning, then output \"Yes\" or \"No\"."
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
