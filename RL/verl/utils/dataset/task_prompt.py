from typing import Dict, Optional


# 原始裸Prompt
SYSTEM_PROMPT_RAW="You are a helpful assistant. Answer the user's question based on the video provided."
QUESTION_TEMPLATE_OPENQA = """{question}"""
def process_question_template_openqa(row):
    prompted_question = QUESTION_TEMPLATE_OPENQA.format(question=row["problem"])
    return prompted_question


SYSTEM_PROMPT_IMAGE_CROP="""You are a helpful assistant. Answer the user's question based on the image provided. Output your thinking process within the <think> and </think> tags. Whenever you find anything unclear, you can zoom in a specific region in the given image to see more clearly by outputing <grounding>{\"bbox_2d\": [x0, y0, x1, y1], \"source\": \"original_image\"}</grounding>, where (x0, y0) and (x1, y1) are the top-left and bottom-right coordinates of the region that you want to zoom in, respectively (suppose the width and height of the image are 1.0), and 'source' refers to the image that you zoom in and could be either 'original_image' or 'observation_i'. Once the final answer is confirmed, put it within <answer> and </answer>."""
QUESTION_TEMPLATE_MCQA = """{question}"""
def process_question_template_mcqa(row):
    prompted_question = QUESTION_TEMPLATE_MCQA.format(question=row["problem"])
    return prompted_question



# 时序定位任务Prompt
SYSTEM_PROMPT_TG="""You are a helpful assistant. Answer the user's question based on the video provided.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps or time ranges.
Then, provide the start and end times within the <answer> </answer> tags.
For example:
<think>Your thinking process here, including analysis with either specific timestamps or time ranges.</think>
<answer>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP]}</answer>
"""
QUESTION_TEMPLATE_TG = """<video>
Please accurately pinpoint the event "{event}" in the video and determine the precise time period of the event.
"""
def process_question_template_tg(row):
    prompted_question = QUESTION_TEMPLATE_TG.format(event=row["problem"])
    return prompted_question



# 多选题原始Prompt
SYSTEM_PROMPT_MCQA_RAW="""You are a helpful assistant. Answer the user's question based on the video provided.
"""
QUESTION_TEMPLATE_MCQA_RAW = """<video>
Answer the question "{question}" according to the content of the video. Select the answer from :{options}.
Answer with the option's letter from the given choices directly, and do not reply with any extra words.
"""
def process_question_template_mcqa_raw(row):
    prompted_question = QUESTION_TEMPLATE_MCQA_RAW.format(question=row["question"], options=str(row["options"]))
    return prompted_question



# 多选题思考Prompt
SYSTEM_PROMPT_MCQA_THINK="""You are a helpful assistant. Answer the user's question based on the video provided.
Output your thought process within the <think> </think> tags, including analysis with either specific timestamps or events.
Then, provide your answer within the <answer> </answer> tags, output the corresponding letter of the option.
"""
QUESTION_TEMPLATE_MCQA_THINK = """<video>
Answer the question "{question}" according to the content of the video. Select the answer from :{options}.
"""
def process_question_template_mcqa_think(row):
    prompted_question = QUESTION_TEMPLATE_MCQA_THINK.format(question=row["question"], options=str(row["options"]))
    return prompted_question



# 多轮视频线索查找工具调用Prompt
SYSTEM_PROMPT_MULTI_ROUND="""You are a helpful assistant. Answer the user's question based on the provided video.
Output your thinking process within the `<think>` and `</think>` tags.
If you find any video segments that might help answer your questions, you can view a specific area in detail by outputting `<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"medium\"}</grounding>`, where t0 and t1 are the start and end times of the video segment you want to observe in detail within the entire video, sampling_strategy must be a string and in ['coarse', 'fine', 'medium'].
Once you believe you have observed all the video clues that help you answer the question and are able to integrate all the existing clues to give the correct answer, you should put the correct answer within the `<answer>` and `</answer>` tags. The final answer should be a single capital letter.
The following is a example of the thinking process and the final answer:
 - When you want to examine a specific clue segment of a video more closely, produce exactly: <think>your thinking process here</think><grounding>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP], "sampling_strategy": "medium"}</grounding>
 - When you believe the current information is sufficient to give an answer, produce exactly: <think>your thinking process here</think><answer>single capital letter ANSWER here</answer>
"""
QUESTION_TEMPLATE_MULTI_ROUND = """<video>
Answer the question: "{question}" according to the content of the video. Select the answer from :{options}.
You are advised to first observe potential clue segments, use <think>YOUR THINK</think><grounding>YOUR GROUNDING</grounding> to to specify the segment you want to observe in detail.
"""
def process_question_template_multi_round(row):
    prompted_question = QUESTION_TEMPLATE_MULTI_ROUND.format(question=row["question"], options=str(row["options"]))
    return prompted_question



SYSTEM_PROMPT_MULTI_ROUND_MC="""You are a helpful assistant. Answer the user's multiple-choice question based on the provided video.
Output your thinking process within the `<think>` and `</think>` tags.
If you find any video segments that might help answer your questions, you can view a specific area in detail by outputting `<grounding>{\"temporal_segment\": [t0, t1], \"sampling_strategy\": \"medium\"}</grounding>`, where t0 and t1 are the start and end times (in integer seconds) of the video segment you want to observe in detail within the entire video, sampling_strategy must be a string and in ['coarse', 'fine', 'medium'].
Once you believe you have observed all the video clues that help you answer the question and are able to integrate all the existing clues to give the correct answer, you should put the correct option letter within the `<answer>` and `</answer>` tags.
The following is a example of the thinking process and the final answer:
 - When you want to examine a specific clue segment of a video more closely, produce exactly: '<think>your thinking process here</think>\n<grounding>{"temporal_segment": [START_TIMESTAMP, END_TIMESTAMP], "sampling_strategy": "medium"}</grounding>\n'.
 - When you believe the current information is sufficient to give an answer, produce exactly: '<think>your thinking process here</think>\n<answer>Option Letter</answer>\n'.
"""
QUESTION_TEMPLATE_MULTI_ROUND_MC = """Here is the original full video (Observation 0): 
<video>
Answer the following multiple-choice question according to the content of the video: "{question}".
Options:
{options} 
You are advised to first observe potential clue segments, use '<think>YOUR THINK</think>\n<grounding>YOUR GROUNDING</grounding>\n' to to specify the segment you want to observe in detail.
If the evidence is visible in the original video for a long enough time and is clear enough to support a confident and correct answer, you may answer directly.
"""
def process_question_template_multi_round_mc(row):
    prompted_question = QUESTION_TEMPLATE_MULTI_ROUND_MC.format(question=row["question"], options=str(row["options"]))
    return prompted_question



SYSTEM_PROMPT_MAP={
    "defaut": SYSTEM_PROMPT_IMAGE_CROP,

    "clue_multi_w_tool": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "clue_multi_wo_tool": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "clue_single_w_tool": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "clue_single_wo_tool": SYSTEM_PROMPT_MULTI_ROUND_MC,
    
    "tool_free_form": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "tool_penalty_single_turn": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "tool_reward_with_round_penalty": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "tool_reward_without_round_penalty": SYSTEM_PROMPT_MULTI_ROUND_MC,

    "temporal_grounded_qa_iou": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "temporal_grounded_qa": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "temporal_grounding": SYSTEM_PROMPT_TG,

    "test_temporal_grounded_qa": SYSTEM_PROMPT_MULTI_ROUND_MC,
    "test_temporal_grounding": SYSTEM_PROMPT_TG,
    
    "charades": SYSTEM_PROMPT_TG,
    "nextgqa": SYSTEM_PROMPT_MULTI_ROUND,
    "mlvu": SYSTEM_PROMPT_MULTI_ROUND,
    "videomme": SYSTEM_PROMPT_MULTI_ROUND,
    "cgbench": SYSTEM_PROMPT_MULTI_ROUND,
    "vrbench": SYSTEM_PROMPT_MULTI_ROUND,
    "videommmu": SYSTEM_PROMPT_MULTI_ROUND,
    "lvbench": SYSTEM_PROMPT_MULTI_ROUND,
    "longvideobench": SYSTEM_PROMPT_MULTI_ROUND,
    "mmvu": SYSTEM_PROMPT_MULTI_ROUND,
    "videoholmes": SYSTEM_PROMPT_MULTI_ROUND,
    "longvideoreason": SYSTEM_PROMPT_MULTI_ROUND,

    # "nextgqa": SYSTEM_PROMPT_MCQA_RAW,
    # "mlvu": SYSTEM_PROMPT_MCQA_RAW,
    # "videomme": SYSTEM_PROMPT_MCQA_RAW,
    # "cgbench": SYSTEM_PROMPT_MCQA_RAW,
    # "vrbench": SYSTEM_PROMPT_MCQA_RAW,
    # "videommmu": SYSTEM_PROMPT_MCQA_RAW,
}

DATA_SOURCE_TEMPLATES_FUNCTION = {
    "default": process_question_template_openqa,

    "clue_multi_w_tool": process_question_template_multi_round_mc,
    "clue_multi_wo_tool": process_question_template_multi_round_mc,
    "clue_single_w_tool": process_question_template_multi_round_mc,
    "clue_single_wo_tool": process_question_template_multi_round_mc,

    "tool_free_form": process_question_template_multi_round_mc,
    "tool_penalty_single_turn": process_question_template_multi_round_mc,
    "tool_reward_with_round_penalty": process_question_template_multi_round_mc,
    "tool_reward_without_round_penalty": process_question_template_multi_round_mc,

    "temporal_grounded_qa_iou": process_question_template_multi_round_mc,
    "temporal_grounded_qa": process_question_template_multi_round_mc,
    "temporal_grounding": process_question_template_tg,
    
    "test_temporal_grounded_qa": process_question_template_multi_round_mc,
    "test_temporal_grounding": process_question_template_tg,

    "charades": process_question_template_tg,
    "nextgqa": process_question_template_multi_round,
    "mlvu": process_question_template_multi_round,
    "videomme": process_question_template_multi_round,
    "cgbench": process_question_template_multi_round,
    "vrbench": process_question_template_multi_round,
    "videommmu": process_question_template_multi_round,
    "lvbench": process_question_template_multi_round,
    "longvideobench": process_question_template_multi_round,
    "mmvu": process_question_template_multi_round,
    "videoholmes": process_question_template_multi_round,
    "longvideoreason": process_question_template_multi_round,

    # "nextgqa": process_question_template_mcqa_raw,
    # "mlvu": process_question_template_mcqa_raw,
    # "videomme": process_question_template_mcqa_raw,
    # "cgbench": process_question_template_mcqa_raw,
    # "vrbench": process_question_template_mcqa_raw,
    # "videommmu": process_question_template_mcqa_raw,
}


def get_system_prompt(row):
    data_source = row.get("data_source", "default")
    if data_source == "default":
        print("<<<Warning>>> Not find data_source in predefined system prompt, use default prompt.")
    # else:
    #     print("<<<Debug>>> func get_system_prompt: data_source is", data_source)
    return SYSTEM_PROMPT_MAP[data_source]


def process_problem_with_data_source(row) -> str:
    data_source = row.get("data_source", "default")
    if data_source == "default":
        print("<<<Warning>>> Not find data_source in predefined question prompt, use default prompt.")
    # else:
    #     print("<<<Debug>>> func process_problem_with_data_source: data_source is", data_source)
    return DATA_SOURCE_TEMPLATES_FUNCTION[data_source](row)






# 中间过程工具调用Prompt
TOOL_CALL_CROP_MULTI_TRUN_PROMPT="After the above Action {action_turn}, here is the the zoom-in image (Observation {observation_turn}):\n<|vision_start|><|image_pad|><|vision_end|>.\nContinue your reasoning process inside <think> and </think>. If needed, you can continue to zoom in on the original image or any of the observations, by outputting <grounding> and </grounding> as before. If the final answer is confirmed, put your final answer inside <answer> and </answer>."


TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT = (
    "After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):\n"
    "<|vision_start|><|video_pad|><|vision_end|>.\n"
    "Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting temporal "
    "segments from the original video by outputting <grounding> and </grounding> as before. Once you are ready "
    "to provide the final answer, put it inside <answer> and </answer>."
)


TOOL_CALL_CROP_VIDEO_MULTI_TRUN_PROMPT_TIME_INSTRUCTION = (
    "After the above Action {action_turn}, here is the refined video clip (Observation {observation_turn}):\n"
    "<|vision_start|><|video_pad|><|vision_end|>{time_instruction}\n"
    "Continue your reasoning process inside <think> and </think>. If needed, you can keep selecting temporal "
    "segments from the original video by outputting <grounding> and </grounding> as before. Once you are ready "
    "to provide the final answer, put it inside <answer> and </answer>."
)


ERROR_INFO_MULTI_TURN_PROMPT="Please analyze the error information obtained from the function tool and adjust your response. Countinue your reasoning process inside <think> and </think>."
