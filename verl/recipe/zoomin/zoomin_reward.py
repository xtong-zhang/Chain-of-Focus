# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
from collections import Counter


def format_reward(predict_str: str) -> float:
    """
    Check if the prediction follows the required format:
    Multiple rounds of <think>reasoning process</think> followed by <answer>answer</answer>
    
    Allows whitespace (spaces, tabs, newlines) between the tags.
    Both <think> and <answer> tags must contain non-empty content.
    """
    predict_str = predict_str.strip()
    predict_str = re.sub(r"<\|im_end\|>\s*$", "", predict_str)

    # 匹配结构：一个或多个<think>...</think> 最后跟一个<answer>...</answer>
    # (?:<think>\s*(.+?)\s*</think>\s*)+  匹配一个或多个think标签
    # <answer>\s*(.+?)\s*</answer>\s*$   匹配最后的answer标签
    pattern = re.compile(
        r"^(?:<think>\s*(.+?)\s*</think>\s*)+<answer>\s*(.+?)\s*</answer>\s*$",
        re.DOTALL
    )
    match_result = pattern.fullmatch(predict_str)

    
    return 1.0 if match_result else 0.0


# def format_reward(text: str) -> float:
#     """
#     检查文本格式是否符合要求
    
#     Args:
#         text: 需要检查的文本字符串
        
#     Returns:
#         bool: 文本格式是否正确
#     """

#     text = text.strip()
#     predict_str = re.sub(r"<\|im_end\|>\s*$", "", text)

#     pattern = re.compile(
#         r"<think>\s*(.+?)\s*</think>\s*<answer>\s*(.+?)\s*</answer>\s*$",
#         re.DOTALL
#     )
#     match_result = pattern.fullmatch(predict_str)
#     if not match_result:
#         return 0.0


#     think_count = text.count("<think>")
#     think_end_count = text.count("</think>")
#     if think_count != 1 or think_end_count != 1:
#         return 0.0
        
#     answer_count = text.count("<answer>")
#     answer_end_count = text.count("</answer>")
#     if answer_count != 1 or answer_end_count != 1:
#         return 0.0
        
#     #  检查image_zoomin和bbox的配对
#     zoomin_pattern = r'<\|image_zoomin\|>'
#     bbox_pattern = r'<\|box_start\|>\[[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+\]<\|box_end\|>'
#     bbox_pattern2 = r'\[[0-9]+,\s*[0-9]+,\s*[0-9]+,\s*[0-9]+\]'

    
#     zoomin_matches = re.findall(zoomin_pattern, text)
#     bbox_matches = re.findall(bbox_pattern, text)
#     bbox2_matches = re.findall(bbox_pattern2, text)

    
#     if len(zoomin_matches) > 0:
#         # 找到所有image_zoomin和bbox的位置
#         zoomin_positions = []
#         bbox_positions = []
        
#         # 收集所有image_zoomin位置
#         start_pos = 0
#         while True:
#             pos = text.find("<|image_zoomin|>", start_pos)
#             if pos == -1:
#                 break
#             zoomin_positions.append(pos)
#             start_pos = pos + 1
            
#         # 收集所有bbox位置
#         start_pos = 0
#         while True:
#             pos = text.find("<|box_start|>", start_pos)
#             if pos == -1:
#                 break
#             bbox_positions.append(pos)
#             start_pos = pos + 1
        
#         # 检查第一个image_zoomin前是否至少有一个bbox
#         if not bbox_positions or bbox_positions[0] >= zoomin_positions[0]:
#             return 0.0
            
#         # 检查每两个连续的image_zoomin之间是否至少有一个bbox
#         for i in range(len(zoomin_positions) - 1):
#             current_zoomin_pos = zoomin_positions[i]
#             next_zoomin_pos = zoomin_positions[i + 1]
            
#             # 检查在这两个zoomin之间是否有bbox
#             has_bbox_between = any(
#                 current_zoomin_pos < bbox_pos < next_zoomin_pos 
#                 for bbox_pos in bbox_positions
#             )
#             if not has_bbox_between:
#                 return 0.0
                
#     #  检查bbox格式是否完整
#     if len(bbox_matches) != len(bbox2_matches) or text.count("<|box_start|>") != text.count("<|box_end|>") or len(bbox_matches) != text.count("<|box_start|>"):
#         return 0.0
    
#     if "<|image_zoomin|><|im_end|>" in text:
#         return 0.0
#     return 1.0

def extract_answer(predict_str: str) -> str:
    """
    Robustly extract the last <answer>...</answer> block, even if it's on a new line or spaced.
    """
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    matches = pattern.findall(predict_str)
    if matches:
        return matches[-1].strip()
    return ""

# def acc_reward(predict_str: str, ground_truth: str) -> float:
#     """
#     Compute accuracy by exact match or substring match.
#     """
#     ground_truth = ground_truth.strip()
#     answer = extract_answer(predict_str).strip()

#     print("--- extract_answer: ", answer)
#     if not answer:
#         return 0.0
    
#     # Exact match
#     if answer.lower() == ground_truth.lower():
#         return 1.0
    
#     # Substring match (either answer is in ground truth or ground truth is in answer)
#     if answer.lower() in ground_truth.lower() or ground_truth.lower() in answer.lower():
#         return 1.0
    
#     return 0.0




def zoom_in_reward(predict_str: str, ground_truth: str) -> float:
    """
    Compute zoom-in reward by counting the number of zoom-in tokens in the answer.
    """
    zoom_in_token = "<tool_call>"
    return float(predict_str.count(zoom_in_token))

def correct_not_zoomin_reward(predict_str: str, ground_truth: str) -> float:
    """
    Compute zoom-in reward by checking if the answer contains the zoom-in token.
    """
    zoom_in_token = "<|image_zoomin|>"
    if zoom_in_token not in predict_str:
        return 1.0
    return 0.0


# from difflib import SequenceMatcher
# from nltk.corpus import wordnet as wn

# if SequenceMatcher(None, a, b).ratio() >= threshold:
#     return True

# syns_a = wn.synsets(a)
# syns_b = wn.synsets(b)
# for syn1 in syns_a:
#     for syn2 in syns_b:
#         sim = syn1.wup_similarity(syn2)
#         if sim and sim > 0.9:
#             return True




def is_similar_token(a: str, b: str, threshold: float = 0.85) -> bool:
    a, b = a.lower(), b.lower()

    if a == b:
        return True

    if a in b or b in a:
        if min(len(a), len(b)) >= 2:
            return True

    return False



def calculate_f1_score(answer_str, ground_truth_str):
    ground_truth = ground_truth_str.strip().lower()

    answer_tokens = answer_str.lower().split()
    ground_truth_tokens = ground_truth.lower().split()

    print("answer_tokens: ", answer_tokens)
    print("ground_truth_tokens: ", ground_truth_tokens)

    if answer_str == "":
        return 0.0

    elif len(answer_tokens) == 1 and len(ground_truth_tokens) == 1:
        pred_token = answer_tokens[0]
        gt_token = ground_truth_tokens[0]
        if is_similar_token(pred_token, gt_token):
            return 1.0
        else:
            return 0.0

    elif answer_str.lower() in ground_truth.lower() or ground_truth.lower() in answer_str.lower():
        return 1.0
    else:
        # 基于词频的 F1 计算
        answer_counter = Counter(answer_tokens)
        ground_truth_counter = Counter(ground_truth_tokens)

        # 计算交集（考虑词频）
        common_tokens = 0
        for token in answer_counter:
            if token in ground_truth_counter:
                common_tokens += min(answer_counter[token], ground_truth_counter[token])

        precision = common_tokens / len(answer_tokens) if answer_tokens else 0
        recall = common_tokens / len(ground_truth_tokens) if ground_truth_tokens else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

        print("f1_score: ", f1_score)
        return f1_score


def f1_acc_score(answer_str: str, ground_truth: str) -> float:
    if calculate_f1_score(answer_str, ground_truth) >= 0.5:
        return 1.0
    else:
        return 0.0


# def zoomin_reward_function(data_source, predict_str: str, ground_truth: str, reward_type: str, extra_info=None) -> float:
def zoomin_reward_function(data_source, solution_str, ground_truth, reward_type:str, extra_info=None):
    """
    Compute the overall score:
    - 90% for answer accuracy
    - 10% for format compliance
    """
    answer_str = extract_answer(solution_str)
    # NOTE: MARK MODIFY
    print("compute_score " + "*" * 100)
    print("reward_type: ", reward_type)
    print("predict_str: ", solution_str)
    print("answer: ", answer_str)
    print("ground_truth: ", ground_truth)

    # acc_score = acc_reward(predict_str, ground_truth)
    acc_score = f1_acc_score(answer_str, ground_truth)
    format_score = format_reward(solution_str)
    zoomin_count = zoom_in_reward(solution_str, ground_truth)


    if zoomin_count > 0:
        zoomin_score = 1
    else:
        zoomin_score = 0


    if reward_type == "acc_format":
        score = 0.9 * acc_score + 0.1 * format_score
        # if zoomin_count > 6:
        #     score = score - 0.2




    print("score: ", score)
    # return {"score": score, "acc_score": acc_score, "format_score": format_score}
    return score

