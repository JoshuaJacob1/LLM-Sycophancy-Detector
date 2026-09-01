import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict

def train_probe(X_train: np.ndarray, y_train: np.ndarray, C: float = 1.0) -> LogisticRegression:
    """
    Train a logistic regression classifier on activations.
    
    Args:
        X_train: Training activations of shape (N, hidden_dim).
        y_train: Training labels of shape (N,).
        C: Regularization parameter.
        
    Returns:
        LogisticRegression: Fitted classifier.
    """
    clf = LogisticRegression(C=C, max_iter=1000, solver='lbfgs', random_state=42)
    clf.fit(X_train, y_train)
    return clf

def evaluate_probe(clf: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate the trained logistic regression probe on test data.
    
    Args:
        clf: Trained LogisticRegression probe.
        X_test: Test activations of shape (N, hidden_dim).
        y_test: Test labels of shape (N,).
        
    Returns:
        Dict[str, float]: Dictionary of evaluation metrics.
    """
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]
    
    return {
        "accuracy": float(accuracy_score(y_test, preds)),
        "auroc": float(roc_auc_score(y_test, probs)),
        "precision": float(precision_score(y_test, preds, zero_division=0.0)),
        "recall": float(recall_score(y_test, preds, zero_division=0.0)),
        "f1": float(f1_score(y_test, preds, zero_division=0.0))
    }

def compare_directions(probe_weights: np.ndarray, dim_direction: np.ndarray) -> float:
    """
    Compute cosine similarity between probe's weight vector and diff-in-means direction.
    
    Args:
        probe_weights: Weights from the logistic regression classifier.
        dim_direction: Diff-in-means direction vector.
        
    Returns:
        float: Cosine similarity value.
    """
    w = probe_weights.flatten()
    d = dim_direction.flatten()
    cos_sim = np.dot(w, d) / (np.linalg.norm(w) * np.linalg.norm(d))
    return float(cos_sim)

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_path: str) -> None:
    """
    Plot and save confusion matrix.
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        output_path: Path to save the generated plot.
    """
    cm = confusion_matrix(y_true, y_pred)
    labels = ["Honest", "Sycophantic"]
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    plt.savefig(output_path)
    plt.close()
