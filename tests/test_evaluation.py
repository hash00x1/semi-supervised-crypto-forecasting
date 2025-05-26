"""
Tests for evaluation utilities
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tempfile
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.evaluation import (
    compute_classification_metrics,
    print_classification_summary,
    plot_metrics_over_iterations,
    compare_models_performance,
    save_results,
    log_memory_usage
)

class TestEvaluationMetrics:
    
    def test_compute_classification_metrics_perfect_prediction(self):
        """Test metrics computation with perfect predictions."""
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 1, 0, 1])
        y_prob = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.95])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        # Perfect prediction should give accuracy = 1.0
        assert metrics['Accuracy'] == 1.0
        assert metrics['F1_Score'] == 1.0
        assert metrics['Precision'] == 1.0
        assert metrics['Recall'] == 1.0
        
        # Check that all required metrics are present
        required_metrics = ['Accuracy', 'F1_Score', 'ROC_AUC', 'PR_AUC', 'Log_Loss',
                           'TP', 'TN', 'FP', 'FN', 'Precision', 'Recall']
        for metric in required_metrics:
            assert metric in metrics
    
    def test_compute_classification_metrics_random_prediction(self):
        """Test metrics computation with balanced random predictions."""
        np.random.seed(42)
        y_true = np.random.choice([0, 1], size=100, p=[0.5, 0.5])
        y_pred = np.random.choice([0, 1], size=100, p=[0.5, 0.5])
        y_prob = np.random.uniform(0, 1, size=100)
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        # Random prediction should give accuracy around 0.5
        assert 0.3 <= metrics['Accuracy'] <= 0.7  # Allow some variance
        assert 0.0 <= metrics['ROC_AUC'] <= 1.0
        assert 0.0 <= metrics['PR_AUC'] <= 1.0
        assert metrics['Log_Loss'] > 0
        
        # Check confusion matrix components
        assert metrics['TP'] + metrics['TN'] + metrics['FP'] + metrics['FN'] == len(y_true)
    
    def test_compute_classification_metrics_edge_cases(self):
        """Test metrics computation with edge cases."""
        # All positive predictions
        y_true = np.array([1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1])
        y_prob = np.array([0.9, 0.8, 0.95, 0.85])
        
        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        
        assert metrics['Accuracy'] == 1.0
        assert metrics['TP'] == 4
        assert metrics['TN'] == 0
        assert metrics['FP'] == 0
        assert metrics['FN'] == 0
    
    def test_plot_metrics_over_iterations(self, temp_directory):
        """Test plotting metrics evolution over iterations."""
        # Create sample iteration data
        iterations_data = []
        for i in range(1, 11):
            iterations_data.append({
                'iteration': i,
                'Accuracy': 0.5 + 0.03 * i + np.random.normal(0, 0.01),
                'ROC_AUC': 0.5 + 0.02 * i + np.random.normal(0, 0.01),
                'PR_AUC': 0.5 + 0.025 * i + np.random.normal(0, 0.01),
                'Log_Loss': 0.7 - 0.02 * i + np.random.normal(0, 0.01)
            })
        
        metrics_df = pd.DataFrame(iterations_data)
        
        # Test plotting (should not raise errors)
        try:
            plot_metrics_over_iterations(metrics_df, save_path=temp_directory)
            # If we get here, plotting succeeded
            assert True
        except Exception as e:
            pytest.fail(f"Plotting failed with error: {e}")
        
        # Check that plot files were created
        plot_files = [f for f in os.listdir(temp_directory) if f.endswith('.png')]
        assert len(plot_files) == 4  # One for each metric
    
    def test_compare_models_performance(self, temp_directory):
        """Test model performance comparison."""
        # Create sample results
        results_dict = {
            'model_1': {
                'Accuracy': 0.85,
                'ROC_AUC': 0.88,
                'PR_AUC': 0.82,
                'F1_Score': 0.84
            },
            'model_2': {
                'Accuracy': 0.82,
                'ROC_AUC': 0.85,
                'PR_AUC': 0.80,
                'F1_Score': 0.81
            },
            'model_3': {
                'Accuracy': 0.88,
                'ROC_AUC': 0.91,
                'PR_AUC': 0.85,
                'F1_Score': 0.87
            }
        }
        
        comparison_df = compare_models_performance(results_dict, save_path=temp_directory)
        
        # Check comparison dataframe
        assert len(comparison_df) == 3
        assert 'Model' in comparison_df.columns
        assert all(model in comparison_df['Model'].values for model in results_dict.keys())
        
        # Check that plot was created
        plot_files = [f for f in os.listdir(temp_directory) if f.endswith('.png')]
        assert len(plot_files) == 1
    
    def test_save_results(self, temp_directory):
        """Test saving results to CSV."""
        # Create sample metrics dataframe
        metrics_data = {
            'iteration': [1, 2, 3],
            'Accuracy': [0.75, 0.78, 0.82],
            'ROC_AUC': [0.77, 0.80, 0.84],
            'F1_Score': [0.73, 0.76, 0.80]
        }
        metrics_df = pd.DataFrame(metrics_data)
        
        # Save results
        file_path = os.path.join(temp_directory, 'test_results.csv')
        save_results(metrics_df, file_path)
        
        # Check that file was created and contains correct data
        assert os.path.exists(file_path)
        
        loaded_df = pd.read_csv(file_path)
        pd.testing.assert_frame_equal(loaded_df, metrics_df)
    
    def test_log_memory_usage(self, capsys):
        """Test memory usage logging."""
        log_memory_usage("Test step")
        
        captured = capsys.readouterr()
        assert "Test step" in captured.out
        assert "RAM:" in captured.out
        assert "GPU:" in captured.out

class TestEvaluationIntegration:
    
    def test_full_evaluation_pipeline(self, temp_directory):
        """Test the complete evaluation pipeline."""
        np.random.seed(42)
        
        # Simulate multiple iterations of training
        all_iterations = []
        for iteration in range(1, 6):
            # Simulate improving performance over iterations
            base_acc = 0.6 + 0.05 * iteration
            y_true = np.random.choice([0, 1], size=100)
            y_pred = np.random.choice([0, 1], size=100, 
                                    p=[1-base_acc, base_acc] if np.random.rand() > 0.5 
                                    else [base_acc, 1-base_acc])
            y_prob = np.random.uniform(0, 1, size=100)
            
            # Compute metrics
            metrics = compute_classification_metrics(y_true, y_pred, y_prob)
            metrics['iteration'] = iteration
            all_iterations.append(metrics)
        
        # Create results dataframe
        results_df = pd.DataFrame(all_iterations)
        
        # Save results
        results_path = os.path.join(temp_directory, 'full_results.csv')
        save_results(results_df, results_path)
        
        # Plot metrics evolution
        plot_metrics_over_iterations(results_df, save_path=temp_directory)
        
        # Verify outputs
        assert os.path.exists(results_path)
        
        # Check that plots were created
        plot_files = [f for f in os.listdir(temp_directory) if f.endswith('.png')]
        assert len(plot_files) >= 4  # At least 4 metric plots
        
        # Verify results can be loaded back
        loaded_results = pd.read_csv(results_path)
        assert len(loaded_results) == 5  # 5 iterations
        assert 'iteration' in loaded_results.columns
        assert 'Accuracy' in loaded_results.columns
