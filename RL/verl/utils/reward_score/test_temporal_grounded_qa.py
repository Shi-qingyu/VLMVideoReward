import re
import json

def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str, use_gpt=False, gpt_extract_answer=False, extra_info=None):

    original_answer = original_answer.strip()
    if isinstance(predict_str_list, list):
        predict_str = predict_str_list[-1].strip()
    else:
        assert isinstance(predict_str_list, str)
        predict_str = predict_str_list.strip()
    
    extract_answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    matches = extract_answer_pattern.findall(predict_str)
    if len(matches) > 0:
        predict_str = matches[-1]
    # else:
    #     return 0.0

    if predict_str.strip() == original_answer:
        return 1.0
    
    pattern = r'^\(([A-Z])\).*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    # match format like "A. xxx"
    pattern = r'^([A-Z])\..*$'
    match = re.match(pattern, predict_str, re.DOTALL)
    if match and match.group(1) == original_answer:
        return 1.0

    return 0.0

def acc_reward(prompt: str, predict_str_list: list, solution: str, extra_info: dict = None) -> float:
    gpt_extract_answer = extra_info.get("gpt_extract_answer", False)
    try:
        reward = inner_acc_reward(prompt, predict_str_list, solution, use_gpt=True, gpt_extract_answer=gpt_extract_answer, extra_info=extra_info)
    except Exception as e:
        print("<<<ERROR>>> Error in inner_acc_reward when processing prompt: ", prompt, "predict_str_list: ", predict_str_list, "solution: ", solution, "error: ", str(e))
        return {"is_filter": True, "info": str(e)}
    return reward


def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    acc = acc_reward(prompt, predict_str_list, ground_truth["answer"], extra_info)
    if isinstance(acc, dict):
        return acc
    acc_score = acc_reward_weight * acc
    format_score = 0.000000001
    score = acc_score
    return score, acc_score, format_score

if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = [
        '<think>Let\'s analyze the video segment around the middle of the video to determine what the girl did. The video shows a girl in a pink jacket standing near a boy who is playing with a toy phone. The girl is seen moving her hands and appears to be interacting with the boy. She seems to be walking away from the boy and towards the camera.</think><grounding>{"temporal_segment": [39.9, 47.9], "sampling_strategy": "medium"}</grounding>', 
        '<think>Let\'s analyze the video segment around the middle of the video to determine what the girl did. The video shows a girl in a pink jacket standing near a boy who is playing with a toy phone. The girl is seen moving her hands and appears to be interacting with the boy. She seems to be walking away from the boy and towards the camera.</think><grounding>{"temporal_segment": [39.9, 47.9], "sampling_strategy": "medium"}</grounding>', 
        '<think>After the girl walks away from the boy and towards the camera, she is seen in the foreground, and the boy is in the background. The girl is wearing a pacifier and appears to be moving her hands, possibly gesturing or talking. The boy is still playing with the toy phone and seems to be engaged in his own activity.</think><answer>A</answer>'
    ]
    predict_str2 = [
        '<think>After the girl walks away from the boy and towards the camera, she is seen in the foreground, and the boy is in the background. The girl is wearing a pacifier and appears to be moving her hands, possibly gesturing or talking. The boy is still playing with the toy phone and seems to be engaged in his own activity.</think><answer>C</answer>'
    ]
    ground_truth = {
        "answer": "C",
        "clue": [
            {
                "timestamp": [
                    13.8,
                    29.8
                ],
                "text": ""
            }
        ]
    }
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 1.0,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)

    s2 = compute_score(question, predict_str2, ground_truth, extra_info)
    print(s2)