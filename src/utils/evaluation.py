"""
Evaluation utilities for model performance assessment.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (classification_report, roc_auc_score, precision_recall_curve, 
                           auc, confusion_matrix, log_loss, accuracy_score, f1_score)
from typing import Dict, List, Any
import psutil
import torch

def log_memory_usage(step_name: str) -> None:
    """Log current memory usage."""
    ram = psutil.virtual_memory().used / (1024 ** 3)  # GB
    gpu = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0  # GB
    print(f"{step_name} - RAM: {ram:.2f}GB, GPU: {gpu:.2f}GB")

def compute_classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                 y_prob: np.ndarray) -> Dict[str, float]:
    """Compute comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_prob)
    logloss = log_loss(y_true, y_prob)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'Accuracy': accuracy,
        'F1_Score': f1,
        'ROC_AUC': roc_auc,
        'PR_AUC': pr_auc,
        'Log_Loss': logloss,
        'TP': tp,
        'TN': tn,
        'FP': fp,
        'FN': fn,
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0
    }

def print_classification_summary(y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model") -> None:
    """Print a comprehensive classification summary."""
    print(f"\n{model_name} Classification Summary:")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=['Fall', 'Rise']))

def plot_metrics_over_iterations(metrics_df: pd.DataFrame, save_path: str = None) -> None:
    """Plot metrics evolution over iterations."""
    plot_metrics = ['Accuracy', 'ROC_AUC', 'PR_AUC', 'Log_Loss']
    
    for metric in plot_metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df['iteration'], metrics_df[metric], marker='o', linewidth=2)
        plt.xlabel('Iteration')
        plt.ylabel(metric)
        plt.title(f'{metric} Evolution Over Iterations')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f'{save_path}/{metric}_evolution.png', dpi=300, bbox_inches='tight')
        plt.show()
        plt.close()

def compare_models_performance(results_dict: Dict[str, Dict[str, Any]], 
                             save_path: str = None) -> pd.DataFrame:
    """Compare performance across different models."""
    comparison_data = []
    
    for model_name, metrics in results_dict.items():
        comparison_data.append({
            'Model': model_name,
            **metrics
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create comparison plots
    metrics_to_plot = ['Accuracy', 'ROC_AUC', 'PR_AUC', 'F1_Score']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, metric in enumerate(metrics_to_plot):
        axes[i].bar(comparison_df['Model'], comparison_df[metric])
        axes[i].set_title(f'{metric} Comparison')
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(f'{save_path}/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return comparison_df

def save_results(metrics_df: pd.DataFrame, file_path: str) -> None:
    """Save results to CSV file."""
    metrics_df.to_csv(file_path, index=False)
    print(f"Results saved to: {file_path}")
