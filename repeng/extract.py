import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from typing import List, Tuple, Any

def load_model(model_name: str = "Qwen/Qwen2.5-1.5B-Instruct") -> Tuple[Any, Any]:
    """
    Loads the model and tokenizer.
    
    Args:
        model_name (str): Name or path of the model.
        
    Returns:
        Tuple[Any, Any]: A tuple of (model, tokenizer).
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16
    ).to(device)
    model.eval()
    
    return model, tokenizer

@torch.no_grad()
def extract_activations(model: Any, tokenizer: Any, texts: List[str], layer_idx: int, batch_size: int = 8, verbose: bool = False) -> np.ndarray:
    """
    Extracts mean-pooled activations from a specific layer.
    
    Args:
        model: The PyTorch model.
        tokenizer: The tokenizer.
        texts (List[str]): List of input texts.
        layer_idx (int): The index of the layer to extract from (0-indexed based on hidden_layers).
        batch_size (int): Batch size for inference.
        verbose (bool): Whether to show a progress bar.
        
    Returns:
        np.ndarray: A numpy array of shape (N, hidden_dim).
    """
    device = get_device(model)
    all_pooled = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting activations", disable=not verbose):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs, output_hidden_states=True)
        # outputs.hidden_states has (num_layers + 1) elements, index 0 is embeddings
        hidden = outputs.hidden_states[layer_idx + 1]
        
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        
        all_pooled.append(pooled.cpu().float().numpy())
        
@torch.no_grad()
def extract_all_layers(model: Any, tokenizer: Any, texts: List[str], batch_size: int = 16, verbose: bool = True) -> List[np.ndarray]:
    """
    Extracts mean-pooled activations for ALL layers in a SINGLE forward pass per batch.
    
    Returns:
        List[np.ndarray]: A list of length num_layers, where each element is an array of shape (N, hidden_dim).
    """
    device = get_device(model)
    num_layers = get_num_layers(model)
    
    # List of lists to hold batch outputs per layer
    layer_batches = [[] for _ in range(num_layers)]
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Extracting all layer activations", disable=not verbose):
        batch_texts = texts[i:i + batch_size]
        
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(device)
        
        outputs = model(**inputs, output_hidden_states=True)
        attention_mask = inputs["attention_mask"]
        mask = attention_mask.unsqueeze(-1).float()
        denom = mask.sum(dim=1).clamp(min=1e-9)
        
        for l in range(num_layers):
            hidden = outputs.hidden_states[l + 1]
            pooled = (hidden * mask).sum(dim=1) / denom
            layer_batches[l].append(pooled.cpu().float().numpy())
            
    return [np.concatenate(batches, axis=0) for batches in layer_batches]

def get_num_layers(model: Any) -> int:
    """
    Returns the number of hidden layers in the model.
    
    Args:
        model: The PyTorch model.
        
    Returns:
        int: Number of hidden layers.
    """
    return model.config.num_hidden_layers

def get_device(model: Any) -> torch.device:
    """
    Returns the device of the model.
    
    Args:
        model: The PyTorch model.
        
    Returns:
        torch.device: The device of the model parameters.
    """
    return next(model.parameters()).device
