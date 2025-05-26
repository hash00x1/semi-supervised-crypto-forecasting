"""
Pseudo-labeling utilities for semi-supervised learning.
"""

import numpy as np
import torch
from typing import Tuple
import gc

def batch_pseudo_labeling_monte_carlo(model: torch.nn.Module, X_unlabeled: np.ndarray, 
                                    batch_size: int = 32, confidence_threshold: float = 0.8,
                                    variance_threshold: float = 0.05, num_mc_samples: int = 3,
                                    device: torch.device = None) -> np.ndarray:
    """Generate pseudo-labels using Monte Carlo sampling."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    pseudo_labels = []
    adjusted_threshold = max(0.5, confidence_threshold)
    rise_threshold = adjusted_threshold - 0.1
    
    with torch.no_grad():
        for i in range(0, len(X_unlabeled), batch_size):
            batch_X = X_unlabeled[i:i + batch_size]
            X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
            
            # Monte Carlo sampling
            preds = [model(X_tensor).cpu().numpy() for _ in range(num_mc_samples)]
            preds_stack = np.stack(preds, axis=0)
            preds_mean = np.mean(preds_stack, axis=0).flatten()
            preds_var = np.var(preds_stack, axis=0).flatten()
            
            # Apply thresholds
            batch_labels = np.where(
                (preds_var < variance_threshold) & (preds_mean > rise_threshold), 1,
                np.where((preds_var < variance_threshold) & (preds_mean < 1 - adjusted_threshold), 0, -1)
            )
            
            pseudo_labels.extend(batch_labels)
            
            # Clean up memory
            del X_tensor, preds, preds_stack, preds_mean, preds_var
            torch.cuda.empty_cache()
    
    return np.array(pseudo_labels)

def batch_pseudo_labeling_regular(model: torch.nn.Module, X_unlabeled: np.ndarray,
                                 batch_size: int = 32, confidence_threshold: float = 0.8,
                                 device: torch.device = None) -> np.ndarray:
    """Generate pseudo-labels using regular prediction."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.eval()
    pseudo_labels = []
    adjusted_threshold = max(0.5, confidence_threshold)
    rise_threshold = adjusted_threshold - 0.1
    
    with torch.no_grad():
        for i in range(0, len(X_unlabeled), batch_size):
            batch_X = X_unlabeled[i:i + batch_size]
            X_tensor = torch.tensor(batch_X, dtype=torch.float32).to(device)
            preds = model(X_tensor).cpu().numpy().flatten()
            
            # Apply thresholds
            batch_labels = np.full_like(preds, -1, dtype=int)
            batch_labels[preds > rise_threshold] = 1
            batch_labels[preds < 1 - adjusted_threshold] = 0
            
            pseudo_labels.extend(batch_labels)
            
            del X_tensor, preds
            torch.cuda.empty_cache()
    
    return np.array(pseudo_labels)

def split_labeled_unlabeled(X: np.ndarray, y: np.ndarray, 
                           labeled_percentage: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into labeled and unlabeled portions."""
    num_sequences = len(X)
    num_labeled = int(labeled_percentage * num_sequences)
    
    X_labeled = X[:num_labeled]
    y_labeled = y[:num_labeled]
    X_unlabeled = X[num_labeled:] if labeled_percentage < 1.0 else np.array([])
    
    return X_labeled, y_labeled, X_unlabeled, np.array([])

def update_training_data(X_labeled: np.ndarray, y_labeled: np.ndarray,
                        X_unlabeled: np.ndarray, pseudo_labels: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Update training data with high-confidence pseudo-labels."""
    high_conf_idx = pseudo_labels != -1
    
    if np.any(high_conf_idx):
        pseudo_X = X_unlabeled[high_conf_idx]
        pseudo_y = pseudo_labels[high_conf_idx]
        
        updated_X = np.concatenate([X_labeled, pseudo_X])
        updated_y = np.concatenate([y_labeled, pseudo_y])
        
        return updated_X, updated_y
    
    return X_labeled, y_labeled
