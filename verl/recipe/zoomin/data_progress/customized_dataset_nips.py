"""
Preprocess a custom JSON dataset to parquet format
"""
import sys
sys.path.insert(0, "/home/zhangxintong/scratch/CODE/verl-4")

import argparse
import os
import json

import pandas as pd
from datasets import Dataset
from PIL import Image
from verl.utils.hdfs_io import copy, makedirs
import random


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", required=True, help="Path to input JSON file")
    parser.add_argument("--local_dir", default="./data/custom_dataset", help="Local directory to save parquet files")
    parser.add_argument("--hdfs_dir", default=None, help="HDFS directory to copy parquet files (optional)")
    parser.add_argument("--split_ratio", default=0.9, type=float, help="Train/test split ratio (default: 0.8)")
    parser.add_argument("--image_root", default="/mnt/buffer/zhangxintong/adaptive_resolution/images_cropped", type=str, help="Image root (default: /mnt/buffer/zhangxintong/adaptive_resolution/images_cropped)")
    args = parser.parse_args()

    # Load JSON data
    with open(args.input_json, 'r') as f:
        json_data = json.load(f)


    random.shuffle(json_data)
    
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
    
    
    instruction_following = (
        r"Please first think through the reasoning process before answering the question. If the relevant area of the image is unclear or difficult to see, first locate the region by generating a bounding box in the following format: <|box_start|>[x1, y1, x2, y2]<|box_end|>. Then, use the <|image_zoomin|> tag to zoom in on that region for closer inspection. Present your reasoning process and answer enclosed within these tags: <think> reasoning process here </think> <answer> answer here </answer>."
    )
    
    # Process dataset
    def make_map_fn(split):
        def process_fn(example, idx):
            # Customize this function based on your JSON structure
            # This is just an example structure similar to gsm8k
            
            # Extract data from the example (adjust field names based on your JSON structure)
            # messages = example["messages"]
            # images = example["images"]
            question = example["question"]
            prompt = "<image>" + instruction_following + "Question: " + question
            data_source = "custom_dataset"
            answer = example["groundtruth"]

            print("answer")
            # only select 1st image
            image_name = example["image_name"]
            root = args.image_root
            image_path = os.path.join(root, image_name)
            image = Image.open(image_path)
            data = {
                "data_source": data_source,
                "agent_name": "tool_agent",
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [image],
                "ability": "vision",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": question,
                },
            }
            print(data)
            return data

        return process_fn
    
    print(1)

    # Process datasets
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

    print(2)

    # Save as parquet files
    os.makedirs(args.local_dir, exist_ok=True)
    train_dataset.to_parquet(os.path.join(args.local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(args.local_dir, "test.parquet"))

    # Upload to HDFS if specified
    if args.hdfs_dir is not None:
        # makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)
        print(f"Files copied to HDFS: {args.hdfs_dir}")
        
    print(f"Conversion complete. Files saved to {args.local_dir}")
