import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from typing import Dict, List, Tuple, Any
from repeng import extract

def compute_direction(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Compute diff-in-means sycophancy direction from activations.
    
    Args:
        X: Activations of shape (N, hidden_dim).
        y: Labels of shape (N,).
        
    Returns:
        np.ndarray: L2-normalized direction vector.
    """
    mu_pos = X[y == 1].mean(axis=0)
    mu_neg = X[y == 0].mean(axis=0)
    diff = mu_pos - mu_neg
    return diff / np.linalg.norm(diff)

def evaluate_direction(X: np.ndarray, y: np.ndarray, direction: np.ndarray) -> Dict[str, float]:
    """
    Project activations onto the given direction and evaluate sycophancy detection.
    
    Args:
        X: Activations of shape (N, hidden_dim).
        y: Labels of shape (N,).
        direction: Direction vector of shape (hidden_dim,).
        
    Returns:
        Dict[str, float]: Dictionary containing AUROC and Accuracy metrics.
    """
    scores = X @ direction
    auroc = float(roc_auc_score(y, scores))
    
    threshold = np.median(scores)
    preds = (scores > threshold).astype(int)
    accuracy = float((preds == y).mean())
    
    return {"auroc": auroc, "accuracy": accuracy}

def run_layer_sweep(model: torch.nn.Module, tokenizer: Any, train_texts: List[str], train_labels: List[int], test_texts: List[str], test_labels: List[int]) -> Tuple[List[Dict], int, np.ndarray]:
    """
    Sweep all layers to compute and evaluate diff-in-means directions.
    
    Args:
        model: Target PyTorch model.
        tokenizer: Model tokenizer.
        train_texts: List of texts for training.
        train_labels: List of integer labels for training.
        test_texts: List of texts for testing.
        test_labels: List of integer labels for testing.
        
    Returns:
        Tuple containing list of results, best layer index, and best direction vector.
    """
    num_layers = extract.get_num_layers(model)
    
    y_train = np.array(train_labels)
    y_test = np.array(test_labels)
    
    results_list = []
    best_auroc = -1.0
    best_layer_idx = 0
    best_direction = None
    
    for l in tqdm(range(num_layers), desc="Sweeping Layers"):
        X_train = extract.extract_activations(model, tokenizer, train_texts, l)
        direction = compute_direction(X_train, y_train)
        
        X_test = extract.extract_activations(model, tokenizer, test_texts, l)
        metrics = evaluate_direction(X_test, y_test, direction)
        
        mu_pos = X_train[y_train == 1].mean(axis=0)
        mu_neg = X_train[y_train == 0].mean(axis=0)
        centroid_distance = float(np.linalg.norm(mu_pos - mu_neg))
        
        results_list.append({
            "layer": l,
            "auroc": metrics["auroc"],
            "accuracy": metrics["accuracy"],
            "centroid_distance": centroid_distance
        })
        
        if metrics["auroc"] > best_auroc:
            best_auroc = metrics["auroc"]
            best_layer_idx = l
            best_direction = direction
            
    return results_list, best_layer_idx, best_direction

def plot_layer_sweep(results: List[Dict], output_path: str) -> None:
    """
    Plot AUROC across different layers and save to output path.
    
    Args:
        results: List of result dictionaries from run_layer_sweep.
        output_path: Path to save the generated plot.
    """
    try:
        plt.style.use('seaborn-v0_8-darkgrid')
    except OSError:
        pass
        
    layers = [r["layer"] for r in results]
    aurocs = [r["auroc"] for r in results]
    
    best_idx = np.argmax(aurocs)
    best_layer = layers[best_idx]
    best_auroc = aurocs[best_idx]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, aurocs, marker='o', linestyle='-')
    plt.plot(best_layer, best_auroc, 'ro', label=f'Peak (Layer {best_layer})')
    
    plt.title("Sycophancy Detection: Layer Sweep")
    plt.xlabel("Layer Index")
    plt.ylabel("AUROC")
    plt.legend()
    plt.grid(True)
    
    plt.savefig(output_path)
    plt.close()
