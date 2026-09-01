import csv
import json
import random
import os
from typing import List, Tuple

def load_contrastive_data(split: str = "train", data_dir: str = "data") -> Tuple[List[str], List[int]]:
    """
    Loads contrastive labeled pairs from JSONL files.
    
    Args:
        split (str): 'train' or 'test'.
        data_dir (str): Directory containing the data.
        
    Returns:
        Tuple[List[str], List[int]]: A tuple of (texts, labels).
    """
    file_path = os.path.join(data_dir, f"contrastive_{split}.jsonl")
    texts = []
    labels = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            texts.append(item["text"])
            labels.append(item["label"])
    return texts, labels

def prepare_data(csv_path: str, output_dir: str, max_train_samples: int = 500, max_test_samples: int = 200) -> None:
    """
    Builds contrastive labeled pairs from the Anthropic sycophancy dataset CSV.
    
    Args:
        csv_path (str): Path to the input CSV file.
        output_dir (str): Directory to save the output JSONL files.
        max_train_samples (int): Max number of training pairs.
        max_test_samples (int): Max number of testing pairs.
    """
    data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question = row.get("question", "")
            match = row.get("answer_matching_behavior", "")
            not_match = row.get("answer_not_matching_behavior", "")
            
            data.append({"text": question + match, "label": 1})
            data.append({"text": question + not_match, "label": 0})
            
    random.seed(42)
    random.shuffle(data)
    
    total_needed = max_train_samples + max_test_samples
    if len(data) > total_needed:
        train_data = data[:max_train_samples]
        test_data = data[max_train_samples:total_needed]
    else:
        split_idx = int(len(data) * 0.8)
        train_data = data[:split_idx]
        test_data = data[split_idx:]
    
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "contrastive_train.jsonl"), "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item) + "\n")
            
    with open(os.path.join(output_dir, "contrastive_test.jsonl"), "w", encoding="utf-8") as f:
        for item in test_data:
            f.write(json.dumps(item) + "\n")
