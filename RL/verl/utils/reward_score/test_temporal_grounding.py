import re
import json

def compute_iou(gt_start, gt_end, pred_start, pred_end):
    intersection = max(0, min(gt_end, pred_end) - max(gt_start, pred_start))
    union = (gt_end - gt_start) + (pred_end - pred_start) - intersection
    return intersection / union

def inner_acc_reward(prompt: str, predict_str_list: list, original_answer: str):
    if isinstance(predict_str_list, list):
        predict_str = predict_str_list[-1].strip()
    else:
        assert isinstance(predict_str_list, str)
        predict_str = predict_str_list.strip()
    
    extract_answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
    matches = extract_answer_pattern.findall(predict_str)
    if len(matches) > 0:
        predict_str = matches[-1]
    else:
        return 0.0
    
    try:
        predict_json = json.loads(predict_str)
        gt_start, gt_end = original_answer
        pred_start, pred_end = predict_json['temporal_segment']
        iou = compute_iou(gt_start, gt_end, pred_start, pred_end)
    except Exception as e:
        print("<<<ERROR>>> Error in temporal_grounding.py --> func inner_acc_reward when calculating iou: ", str(e),"when processing predict_str: ", predict_str)
        return 0.0
    return iou

def acc_reward(prompt: str, predict_str_list: list, solution: str) -> float:
    try:
        reward = inner_acc_reward(prompt, predict_str_list, solution)
    except Exception as e:
        print("<<<ERROR>>> Error in temporal_grounding.py --> func acc_reward when processing prompt: ", prompt, "predict_str_list: ", predict_str_list, "solution: ", solution, "error: ", str(e))
        return {"is_filter": True, "info": str(e)}
    return reward



def grounding_format_reward(predict_str_list: list):
    if isinstance(predict_str_list, list):
        if len(predict_str_list) == 0:
            return 0.0, 0
        predict_str = predict_str_list[-1]
    else:
        predict_str = predict_str_list

    if not isinstance(predict_str, str):
        return 0.0, 0

    pattern = re.compile(r'^(\n?)<think>(.*?)</think>(\n?)<answer>(.*?)</answer>(\n?)$', re.DOTALL)
    match = pattern.match(predict_str)

    if match:
        answer_content = match.group(4)
        try:
            data = json.loads(answer_content)
            if "temporal_segment" in data and len(data["temporal_segment"]) == 2:
                return 1.0, 0
        except:
            pass
    
    return 0.0, 0



def compute_score(prompt: str, predict_str_list: list, ground_truth: list, extra_info: dict = None) -> float:
    acc_reward_weight = extra_info.get('acc_reward_weight', 1.0) if extra_info else 1.0
    format_reward_weight = extra_info.get('format_reward_weight', 1.0) if extra_info else 1.0

    acc = acc_reward(prompt, predict_str_list, ground_truth["clue"][0]["timestamp"])
    if isinstance(acc, dict):
        return acc

    acc_score = acc_reward_weight * acc
    format_score = 0.0000000001    # 与训练中的0格式奖励区分开，防止混淆训练与测试

    score = acc_score 
    return score, acc_score, format_score


if __name__ == '__main__':
    question = "Elena Ferrante" #"<image>\nHint: Please answer the question and provide the final answer at the end.\nQuestion: How many states are represented by the lightest color on the map?" #"<image>What is the output score when the first input is 4 and the second input is 5 according to the Hamlet Evaluation System shown in Figure 2?" #"<image>Who wrote this book?\nAnswer the question with a short phrase."
    predict_str = ["""<think>To determine the name of the store with a blue sign</think><answer>{"temporal_segment": [20.0,     30.0]}</answer>"""]
    ground_truth = {
        "answer": "",
        "clue": [
            {
                "timestamp": [
                    20.8,
                    30.0
                ],
                "text": ""
            }
        ]
    } 
    extra_info = {
        "acc_reward_weight": 1.0,
        "format_reward_weight": 0.5,
        "use_tool_reward_weight": 0.5,
        "gpt_extract_answer": True,
        "extract_answer_tags": "strict",
    }
    s1 = compute_score(question, predict_str, ground_truth, extra_info)
    print(s1)