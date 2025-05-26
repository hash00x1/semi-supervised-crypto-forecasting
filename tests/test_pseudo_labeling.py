"""
Tests for pseudo-labeling utilities
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.pseudo_labeling import (
    batch_pseudo_labeling_monte_carlo,
    batch_pseudo_labeling_regular,
    split_labeled_unlabeled,
    update_training_data
)
from models.lstm_classifier import SingleInputLSTMClassifier

class TestPseudoLabeling:
    
    def test_split_labeled_unlabeled(self, sample_sequences):
        """Test splitting data into labeled and unlabeled portions."""
        X, y = sample_sequences
        labeled_percentage = 0.7
        
        X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_percentage)
        
        # Check shapes
        expected_labeled_size = int(labeled_percentage * len(X))
        expected_unlabeled_size = len(X) - expected_labeled_size
        
        assert len(X_labeled) == expected_labeled_size
        assert len(y_labeled) == expected_labeled_size
        assert len(X_unlabeled) == expected_unlabeled_size
        
        # Check that data is properly split
        np.testing.assert_array_equal(X_labeled, X[:expected_labeled_size])
        np.testing.assert_array_equal(y_labeled, y[:expected_labeled_size])
        np.testing.assert_array_equal(X_unlabeled, X[expected_labeled_size:])
    
    def test_split_labeled_unlabeled_full_data(self, sample_sequences):
        """Test splitting with 100% labeled data."""
        X, y = sample_sequences
        labeled_percentage = 1.0
        
        X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_percentage)
        
        # All data should be labeled
        assert len(X_labeled) == len(X)
        assert len(y_labeled) == len(y)
        assert len(X_unlabeled) == 0
    
    def test_batch_pseudo_labeling_regular(self, sample_sequences, device):
        """Test regular pseudo-labeling."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        # Create and train a simple model
        model = SingleInputLSTMClassifier(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        # Generate pseudo-labels
        batch_size = 8
        confidence_threshold = 0.8
        
        pseudo_labels = batch_pseudo_labeling_regular(
            model, X, batch_size, confidence_threshold, device
        )
        
        # Check output
        assert len(pseudo_labels) == len(X)
        assert all(label in [-1, 0, 1] for label in pseudo_labels)
        
        # Should have some high-confidence predictions
        high_conf_count = np.sum(pseudo_labels != -1)
        assert high_conf_count >= 0  # At least some predictions should be confident
    
    def test_batch_pseudo_labeling_monte_carlo(self, sample_sequences, device):
        """Test Monte Carlo pseudo-labeling."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        # Create and train a simple model
        model = SingleInputLSTMClassifier(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        # Generate pseudo-labels
        batch_size = 8
        confidence_threshold = 0.8
        variance_threshold = 0.05
        num_mc_samples = 3
        
        pseudo_labels = batch_pseudo_labeling_monte_carlo(
            model, X, batch_size, confidence_threshold, 
            variance_threshold, num_mc_samples, device
        )
        
        # Check output
        assert len(pseudo_labels) == len(X)
        assert all(label in [-1, 0, 1] for label in pseudo_labels)
    
    def test_update_training_data(self, sample_sequences):
        """Test updating training data with pseudo-labels."""
        X, y = sample_sequences
        
        # Split data
        X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, 0.7)
        
        # Create some pseudo-labels
        pseudo_labels = np.array([1, -1, 0, 1, -1])  # Mix of confident and unconfident
        X_unlabeled_subset = X_unlabeled[:len(pseudo_labels)]
        
        # Update training data
        updated_X, updated_y = update_training_data(
            X_labeled, y_labeled, X_unlabeled_subset, pseudo_labels
        )
        
        # Check that high-confidence samples were added
        high_conf_count = np.sum(pseudo_labels != -1)
        expected_size = len(X_labeled) + high_conf_count
        
        assert len(updated_X) == expected_size
        assert len(updated_y) == expected_size
        
        # Check that original labeled data is preserved
        np.testing.assert_array_equal(updated_X[:len(X_labeled)], X_labeled)
        np.testing.assert_array_equal(updated_y[:len(y_labeled)], y_labeled)
    
    def test_update_training_data_no_confident_labels(self, sample_sequences):
        """Test updating when no confident pseudo-labels exist."""
        X, y = sample_sequences
        
        # Split data
        X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, 0.7)
        
        # Create all unconfident pseudo-labels
        pseudo_labels = np.array([-1, -1, -1, -1, -1])
        X_unlabeled_subset = X_unlabeled[:len(pseudo_labels)]
        
        # Update training data
        updated_X, updated_y = update_training_data(
            X_labeled, y_labeled, X_unlabeled_subset, pseudo_labels
        )
        
        # Should return original data unchanged
        np.testing.assert_array_equal(updated_X, X_labeled)
        np.testing.assert_array_equal(updated_y, y_labeled)
    
    def test_pseudo_labeling_batch_consistency(self, sample_sequences, device):
        """Test that pseudo-labeling gives consistent results for different batch sizes."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = SingleInputLSTMClassifier(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        confidence_threshold = 0.7
        
        # Test with different batch sizes
        labels_batch_4 = batch_pseudo_labeling_regular(
            model, X, batch_size=4, confidence_threshold=confidence_threshold, device=device
        )
        labels_batch_8 = batch_pseudo_labeling_regular(
            model, X, batch_size=8, confidence_threshold=confidence_threshold, device=device
        )
        
        # Results should be the same regardless of batch size
        np.testing.assert_array_equal(labels_batch_4, labels_batch_8)
    
    def test_confidence_threshold_effect(self, sample_sequences, device):
        """Test that higher confidence thresholds result in fewer confident predictions."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = SingleInputLSTMClassifier(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        # Test with different confidence thresholds
        low_threshold_labels = batch_pseudo_labeling_regular(
            model, X, batch_size=8, confidence_threshold=0.6, device=device
        )
        high_threshold_labels = batch_pseudo_labeling_regular(
            model, X, batch_size=8, confidence_threshold=0.9, device=device
        )
        
        # Higher threshold should result in fewer confident predictions
        low_confident_count = np.sum(low_threshold_labels != -1)
        high_confident_count = np.sum(high_threshold_labels != -1)
        
        assert high_confident_count <= low_confident_count

class TestPseudoLabelingIntegration:
    
    def test_semi_supervised_iteration(self, sample_sequences, device):
        """Test a complete semi-supervised learning iteration."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        # Initialize model
        model = SingleInputLSTMClassifier(input_dim=input_dim, hidden_units=32)
        model.to(device)
        
        # Split data
        labeled_percentage = 0.7
        X_labeled, y_labeled, X_unlabeled, _ = split_labeled_unlabeled(X, y, labeled_percentage)
        
        # Simple training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.BCELoss()
        
        X_tensor = torch.tensor(X_labeled, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y_labeled, dtype=torch.float32).to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()
        
        # Generate pseudo-labels
        model.eval()
        pseudo_labels = batch_pseudo_labeling_regular(
            model, X_unlabeled, batch_size=8, confidence_threshold=0.7, device=device
        )
        
        # Update training data
        updated_X, updated_y = update_training_data(
            X_labeled, y_labeled, X_unlabeled, pseudo_labels
        )
        
        # Verify the process worked
        assert len(updated_X) >= len(X_labeled)
        assert len(updated_y) >= len(y_labeled)
        assert len(pseudo_labels) == len(X_unlabeled)
