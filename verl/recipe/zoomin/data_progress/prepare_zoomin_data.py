
import argparse
import os
import json

# import datasets
import pandas as pd
from datasets import Dataset
import random
from PIL import Image

from verl.utils.hdfs_io import copy, makedirs


SYSTEM_PROMPT = """You are a helpful assistant.

# Tools
You may call one or more functions to assist with the user query.
You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name":"image_zoom_in_tool","description":"Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) and an optional object label.","parameters":{"properties":{"bbox_2d":{"type":"array","items":{"type":"number"},"minItems":4,"maxItems":4,"description":"The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner."},"label":{"type":"string","description":"The name or label of the object in the specified bounding box (optional)."}},"required":["bbox_2d"], "type":"object"},"args_format": "Format the arguments as a JSON object."}}
</tools>

For the function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""

USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed)."

# USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...</answer> (if no tools needed). The content inside <answer> must be a single word or phrase."
# USER_PROMPT = "\nThink in the mind first, and then decide whether to call tools one or more times OR provide final answer. Format strictly as: <think>...</think> <tool_call>...</tool_call> <tool_call>...</tool_call> (if any tools needed) OR <answer>...a single word or phrase...</answer> (if no tools needed)."




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--local_dir", default="~/data/zoomin_multiturn_w_tool")
    parser.add_argument("--hdfs_dir", default=None)
    parser.add_argument("--image_root", default="/mnt/buffer/zhangxintong/adaptive_resolution/images_cropped", type=str, help="Image root (default: /mnt/buffer/zhangxintong/adaptive_resolution/images_cropped)")
    parser.add_argument("--split_ratio", default=0.9, type=float, help="Train/test split ratio (default: 0.8)")

    args = parser.parse_args()


    # Load JSON data
    with open(args.input_json, 'r') as f:
        json_data = json.load(f)



     # Convert to Dataset format
    if isinstance(json_data, list):
        # If JSON is a list of items
        dataset = Dataset.from_list(json_data)
    elif isinstance(json_data, dict):
        # If JSON has a specific structure, adjust as needed
        dataset = Dataset.from_dict(json_data)
    else:
        raise ValueError("Unsupported JSON format")
    

    # Create train/test split
    dataset = dataset.shuffle(seed=42)
    split_datasets = dataset.train_test_split(train_size=args.split_ratio)
    
    train_dataset = split_datasets["train"]
    test_dataset = split_datasets["test"]
    



    # data_source = "hiyouga/geometry3k"
    # dataset = datasets.load_dataset(data_source)
    # train_dataset = dataset["train"]
    # test_dataset = dataset["test"]
    # instruction_following = (
    #     r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
    #     r"The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."
    # )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            question = example.pop("question")
            # prompt = "<image>" + "Question: " + question + USER_PROMPT
            groundtruth = example.pop("groundtruth")
            if len(groundtruth) == 1 and groundtruth.lower() in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l'] and (('(A)' in question and '(B)' in question) or ('A.' in question and 'B.' in question)):
                prompt = "<image>" + " Question: " + question + "\nAnswer with the option's letter from the given choices directly." + USER_PROMPT

            else:
                prompt = "<image>" + " Question: " + question + "\nAnswer the question using a single word or phrase." + USER_PROMPT
            
            print("********** prompt: ", prompt)


            # prompt = "<image>" + "Question: " + question + USER_PROMPT
            image_name = example.pop("image_name")
            root = args.image_root
            image_path = os.path.join(root, image_name)
            image = Image.open(image_path)

            # 补充一下
            data_source = example.pop("data_source", "rl_multiresolution")
            if data_source is None:
                data_source = "unknown"
            print(f"********** data_source: {data_source}")

            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                "images": [{"image": f"file://{image_path}"}], # Based on modifications from from qwen_vl_utils import fetch_image, fetch_video, but when data is saved to parquet files (lines 134-135), PIL Image objects cannot be directly serialized to parquet format, so the datasets library automatically converts them to dictionary format.
                "ability": "vision",
                "reward_model": {"style": "rule", "ground_truth": groundtruth},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": groundtruth,
                    "question": question,
                    "need_tools_kwargs": True,
                    "tools_kwargs": {
                        "image_zoom_in_tool": {
                            "create_kwargs": {"question": question, "image_path": image_path, "ground_truth": groundtruth},
                            "execute_kwargs": {"image_path": image_path},
                            # "calc_reward_kwargs": {},
                            # "release_kwargs": {},
                        },
                    },
                },
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)
    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))
    if hdfs_dir is not None:
        # makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)
