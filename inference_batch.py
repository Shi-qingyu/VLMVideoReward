from transformers import AutoModelForImageTextToText, AutoProcessor

# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen3-VL-2B-Instruct", dtype="auto", device_map="auto"
)

processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct")

# for batch generation, padding_side should be set to left!
processor.tokenizer.padding_side = 'left'

# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "image", "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
    {"role": "user", "content": [{"type": "text", "text": "Who are you?"}]},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
    padding=True # padding should be set for batch generation!
)
inputs = inputs.to(model.device)

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)