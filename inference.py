from transformers import AutoModelForImageTextToText, AutoProcessor
from src.train.checkpoint_utils import prepare_inference_model_dir


model_path = "output/qwen3vl-2b-baseline-1e-bs4-ga4-fps1-t-polished-v3-457"
inference_model_path = prepare_inference_model_dir(model_path)
# default: Load the model on the available device(s)
model = AutoModelForImageTextToText.from_pretrained(
    inference_model_path,
    dtype="auto", 
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(inference_model_path)

question = (
    "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\n"
    "Evaluate the video according to the following dimensions.\n"
    "Video Quality: whether the video is free from major visual defects, including blur, lack of detail, "
    "poor texture, lighting issues, color distortion, flickering, and overexposure.\n"
    "Subject Movement: whether the subject's motion is natural, smooth, and realistic.\n"
    "Physical Interaction: whether interactions among subjects and/or objects are physically plausible.\n"
    "Cause-Effect: whether causal relationships are correctly depicted.\n"
    "Subject Existence: whether the subject described in the prompt appears and is accurate.\n"
    "Object Existence: whether the object described in the prompt appears and is accurate.\n"
    "Subject-Object Interaction: whether the interaction described in the prompt is correctly represented.\n"
    "Prompt: {prompt} Provide your reasoning trace between think tags <think> and </think>, "
    "then output \"Yes\" or \"No\" for each dimension between <answer> and </answer>."
)

# question = (
#     "Suppose you are an expert in judging and evaluating the quality of AI-generated videos.\n"
#     "Evaluate the video according to the following dimensions.\n"
#     "Video Quality: whether the video is free from major visual defects, including blur, lack of detail, "
#     "poor texture, lighting issues, color distortion, flickering, and overexposure.\n"
#     "Motion & Interaction: whether the subject's motion is natural, smooth, and realistic; "
#     "whether interactions among subjects and/or objects are physically plausible; "
#     "and whether causal relationships are correctly depicted.\n"
#     "Prompt Alignment: whether the subject and object described in the prompt appear accurately, "
#     "and whether the subject-object interaction described in the prompt is correctly represented.\n"
#     "Prompt: {prompt} Provide your reasoning trace between think tags <think> and </think>, "
#     "then output \"Yes\" or \"No\" for each dimension between <answer> and </answer>."
# )

prompt = "A Black man in a short-sleeve shirt stands at a kitchen stove. He holds a box of dry pasta in one hand and pours the pasta into a pot of boiling water. Using a wooden spoon, he gently presses the pasta down to ensure it is fully submerged. The background shows kitchen counters and utensils. The camera remains steady, focusing on the man’s hands and the pot."

user_input = question.format(prompt=prompt)

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
                "text": user_input
            },
        ],
    }
]

sample_fps = processor.video_processor.fps
temporal_patch_size = processor.video_processor.temporal_patch_size
videos = messages[0]["content"][0].get("video")
if isinstance(videos, str):
    videos = [videos]

# Build media pools with absolute paths
vp_output = processor.video_processor(videos=videos, return_metadata=True)
video_metadata = vp_output.video_metadata[0]
video_grid_thw = vp_output.video_grid_thw

total_frames = int(video_grid_thw[0][0] * temporal_patch_size)
duration = video_metadata["duration"]
time_instruction = (
    f"This video is uniformly sampled at {sample_fps:.2f} fps, contains {total_frames} frames "
    f"from 0 seconds to {duration:.1f} seconds."
)
original_text = messages[0]["content"][1]["text"]
messages[0]["content"][1]["text"] = f"{time_instruction}\n{original_text}"

# Preparation for inference
inputs = processor.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
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
