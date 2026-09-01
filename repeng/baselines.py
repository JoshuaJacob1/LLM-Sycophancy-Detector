import csv
import torch
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from typing import Dict, Any, List

def random_baseline(y_true: np.ndarray, n_trials: int = 100, seed: int = 42) -> Dict[str, float]:
    """
    Computes baseline metrics using uniform random predictions.
    
    Args:
        y_true (np.ndarray): True labels.
        n_trials (int): Number of random trials.
        seed (int): Random seed.
        
    Returns:
        Dict[str, float]: Dictionary containing mean accuracy, AUROC, and F1.
    """
    np.random.seed(seed)
    
    accuracies = []
    aurocs = []
    f1s = []
    
    for _ in range(n_trials):
        y_pred_prob = np.random.uniform(size=len(y_true))
        y_pred = (y_pred_prob > 0.5).astype(int)
        
        accuracies.append(accuracy_score(y_true, y_pred))
        
        # Handle case where all y_true might be same class
        if len(np.unique(y_true)) > 1:
            aurocs.append(roc_auc_score(y_true, y_pred_prob))
        else:
            aurocs.append(0.5)
            
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
        
    return {
        "accuracy": float(np.mean(accuracies)),
        "auroc": float(np.mean(aurocs)),
        "f1": float(np.mean(f1s))
    }

@torch.no_grad()
def measure_sycophancy_rate(model: Any, tokenizer: Any, csv_path: str, max_samples: int = 200) -> Dict[str, Any]:
    """
    Measures how often the model chooses the sycophantic option.
    
    Args:
        model: The PyTorch model.
        tokenizer: The tokenizer.
        csv_path (str): Path to the Anthropic CSV.
        max_samples (int): Maximum number of samples to evaluate.
        
    Returns:
        Dict[str, Any]: Dictionary containing sycophancy_rate, total, and sycophantic_picks.
    """
    device = next(model.parameters()).device
    
    token_a = tokenizer.encode(" A", add_special_tokens=False)[-1]
    token_b = tokenizer.encode(" B", add_special_tokens=False)[-1]
    token_a_no_space = tokenizer.encode("A", add_special_tokens=False)[-1]
    token_b_no_space = tokenizer.encode("B", add_special_tokens=False)[-1]
    
    sycophantic_picks = 0
    total = 0
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if total >= max_samples:
                break
                
            question = row.get("question", "")
            match = row.get("answer_matching_behavior", "").strip()
            
            # The prompt is just the question
            inputs = tokenizer(question, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # Get logits of the last token
            next_token_logits = outputs.logits[0, -1, :]
            
            prob_a = max(next_token_logits[token_a].item(), next_token_logits[token_a_no_space].item())
            prob_b = max(next_token_logits[token_b].item(), next_token_logits[token_b_no_space].item())
            
            model_pick = "(A)" if prob_a > prob_b else "(B)"
            
            if match.startswith(model_pick):
                sycophantic_picks += 1
                
            total += 1
            
    rate = (sycophantic_picks / total * 100.0) if total > 0 else 0.0
    
    return {
        "sycophancy_rate": rate,
        "total": total,
        "sycophantic_picks": sycophantic_picks
    }
