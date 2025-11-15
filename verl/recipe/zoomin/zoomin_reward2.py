import re
# import evaluate
import string
from collections import Counter


# exact_match = evaluate.load("exact_match")


def normalize_answer(s):
#     # def remove_articles(text):
#     #     return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    # def remove_punc(text):
    #     return "".join(
    #         ch if not unicodedata.category(ch).startswith("P") else " "
    #         for ch in text
    #     )

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace("_", " ")

    # return remove_punc(lower(replace_underscore(s)))
    return white_space_fix(lower(remove_punc(replace_underscore(s))))


# def bool_mapping(s):
#     if s == "True" or s == "true":
#         return "yes"
#     elif s == "False" or s == "false":
#         return "no"
#     else:
#         return s


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    # if (
    #     normalized_prediction in ["yes", "no", "noanswer"]
    #     and normalized_prediction != normalized_ground_truth
    # ):
    #     return ZERO_METRIC
    # if (
    #     normalized_ground_truth in ["yes", "no", "noanswer"]
    #     and normalized_prediction != normalized_ground_truth
    # ):
    #     return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())



    print("prediction: ", prediction)
    print("ground_truth: ", ground_truth)
    print("normalized_prediction: ", normalized_prediction)
    print("normalized_ground_truth: ", normalized_ground_truth)
    print("prediction_tokens: ", prediction_tokens)
    print("ground_truth_tokens: ", ground_truth_tokens)

    print("num_same: ", num_same)
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2.0 * precision * recall) / (precision + recall)


   
    print("f1: ", f1)
    print("precision: ", precision)
    print("recall: ", recall)
    return f1, precision, recall

def extract_answer(predict_str: str) -> str:
    """
    Robustly extract the last <answer>...</answer> block, even if it's on a new line or spaced.
    """
    pattern = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL)
    matches = pattern.findall(predict_str)
    if matches:
        return matches[-1].strip()
    return ""

    
# def compute_score(predict_str: str, ground_truth: str) -> float:
def zoomin_reward_function(data_source, solution_str, ground_truth, reward_type:str, extra_info=None):
    print("compute_score " + "*" * 100)
    print("reward_type: ", reward_type)
    print("solution_str: ", solution_str)
    print("ground_truth: ", ground_truth)

    is_format_error = False
    predict_str = solution_str
    # count_1 = predict_str.count("<|begin_of_documents|>\n")
    # count_2 = predict_str.count("<|end_of_documents|>\n")
    # count_3 = predict_str.count("<|begin_of_query|>")
    # count_4 = predict_str.count("<|end_of_query|>")
    # count_5 = predict_str.count("<|begin_of_documents|>")
    # count_6 = predict_str.count("<|end_of_documents|>")
    # count_7 = predict_str.count("<|begin_of_documents|>\n(1)")
    # if not (count_1 == count_2 == count_3 == count_4 == count_5 == count_6 == count_7):
    #     is_format_error = True

    # count_assiatant_1 = predict_str.count("Assistant:")
    # count_assiatant_2 = predict_str.count("assistant:")
    # if count_assiatant_1 != 0 or count_assiatant_2 != 0:
    #     is_format_error = True

    count_think_1 = predict_str.count("<think>")
    count_think_2 = predict_str.count("</think>")
    if count_think_1 != count_think_2 or count_think_1 == 0 or count_think_2 == 0:
        print("format error: count_think_1 != count_think_2 or count_think_1 == 0")
        is_format_error = True

    count_answer_1 = predict_str.count("<answer>")
    count_answer_2 = predict_str.count("</answer>")
    if count_answer_1 != count_answer_2 or count_answer_1 == 0 or count_answer_2 == 0:
        print("format error: count_answer_1 != 1 or count_answer_2 != 1 or count_answer_1 == 0")
        is_format_error = True

    answer_text = extract_answer(predict_str)
    # if "begin_of_query" in answer_text or "begin_of_documents" in answer_text:
    #     is_format_error = True

    answer_len = len(answer_text.split())
    if answer_len > 10:
        print("format error: answer_len > 10")
        is_format_error = True

    # if count_7 == 0:
    #     is_format_error = True

    # retrieval_pattern = re.compile(r'<\|begin_of_query\|>(.*?)<\|end_of_query\|>', re.DOTALL)
    # retrieval_match = re.search(retrieval_pattern, predict_str)
    # doc_pattern = re.compile(r'<\|begin_of_documents\|>(.*?)<\|end_of_documents\|>', re.DOTALL)
    # doc_match = re.search(doc_pattern, predict_str)

    # retrieval_reward = 1.0 if count_7 >= 1 else -1.0
    # em_score = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
    acc_reward, _ , _ = f1_score(answer_text, ground_truth)
    # acc_reward = 2.0 * acc_reward
    if acc_reward > 0.5:
        acc_reward = 1.0
    else:
        acc_reward = 0.0
    

    format_reward = 0.0 if is_format_error else 1.0
    # return format_reward + retrieval_reward + acc_reward
    

    # return format_reward + acc_reward
    score = 0.2 * format_reward + 0.8 * acc_reward
    print("format_reward: ", format_reward)
    print("acc_reward: ", acc_reward)
    print("score: ", score)
    
    return {"score": score, "format_score": format_reward, "acc_score": acc_reward}




# def compute_score_eval(predict_str: str, ground_truth: str) -> float:
#     predict_no_think = predict_str.split('</think>')[-1].strip()
#     answer_text = predict_no_think.split("<answer>")[-1].split("</answer>")[0].strip()
#     score_info = exact_match.compute(references=[ground_truth], predictions=[answer_text], ignore_case=True, ignore_punctuation=True)
#     acc_reward = float(score_info['exact_match'])
#     return acc_reward