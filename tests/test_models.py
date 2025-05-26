"""
Tests for LSTM models
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.lstm_classifier import SingleInputLSTMClassifier as RegularLSTM
from models.monte_carlo_attention import SingleInputLSTMClassifier as MonteCarloLSTM

class TestLSTMModels:
    
    def test_regular_lstm_initialization(self):
        """Test regular LSTM model initialization."""
        input_dim = 25
        hidden_units = 64
        
        model = RegularLSTM(input_dim=input_dim, hidden_units=hidden_units)
        
        # Check model structure
        assert model.hidden_units == hidden_units
        assert isinstance(model.lstm1, nn.LSTM)
        assert isinstance(model.lstm2, nn.LSTM)
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.sigmoid, nn.Sigmoid)
        
        # Check LSTM configurations
        assert model.lstm1.input_size == input_dim
        assert model.lstm1.hidden_size == hidden_units
        assert model.lstm1.bidirectional == True
        
        # Check final layer dimensions
        assert model.fc.in_features == hidden_units * 2  # Bidirectional
        assert model.fc.out_features == 1
    
    def test_monte_carlo_lstm_initialization(self):
        """Test Monte Carlo LSTM model initialization."""
        input_dim = 25
        hidden_units = 64
        
        model = MonteCarloLSTM(input_dim=input_dim, hidden_units=hidden_units)
        
        # Check model structure
        assert model.hidden_units == hidden_units
        assert isinstance(model.lstm1, nn.LSTM)
        assert isinstance(model.lstm2, nn.LSTM)
        assert hasattr(model, 'attention')
        assert isinstance(model.fc, nn.Linear)
        assert isinstance(model.sigmoid, nn.Sigmoid)
    
    def test_regular_lstm_forward_pass(self, sample_sequences, device):
        """Test regular LSTM forward pass."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = RegularLSTM(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)
        
        # Check output shape and range
        assert output.shape == (X.shape[0], 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_monte_carlo_lstm_forward_pass(self, sample_sequences, device):
        """Test Monte Carlo LSTM forward pass."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = MonteCarloLSTM(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        # Forward pass
        with torch.no_grad():
            output = model(X_tensor)
        
        # Check output shape and range
        assert output.shape == (X.shape[0], 1)
        assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output
    
    def test_model_training_step(self, sample_sequences, device):
        """Test that models can perform a training step."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        for ModelClass in [RegularLSTM, MonteCarloLSTM]:
            model = ModelClass(input_dim=input_dim, hidden_units=32)
            model.to(device)
            model.train()
            
            # Convert to tensors
            X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
            
            # Setup training
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            # Training step
            optimizer.zero_grad()
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            loss.backward()
            optimizer.step()
            
            # Check that loss is computed and gradients exist
            assert not torch.isnan(loss)
            assert loss.item() > 0
            
            # Check that gradients were computed
            for param in model.parameters():
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any()
    
    def test_model_gradient_flow(self, sample_sequences, device):
        """Test that gradients flow properly through the model."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = RegularLSTM(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.train()
        
        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(device).unsqueeze(1)
        
        # Forward and backward pass
        output = model(X_tensor)
        loss = nn.BCELoss()(output, y_tensor)
        loss.backward()
        
        # Check that all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
    
    def test_model_deterministic_output(self, sample_sequences, device):
        """Test that models produce deterministic output in eval mode."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        model = RegularLSTM(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            output1 = model(X_tensor)
            output2 = model(X_tensor)
        
        # Outputs should be identical in eval mode
        torch.testing.assert_close(output1, output2)
    
    def test_different_input_sizes(self, device):
        """Test models with different input dimensions."""
        input_dims = [10, 25, 50]
        batch_size = 8
        sequence_length = 30
        
        for input_dim in input_dims:
            X = torch.randn(batch_size, sequence_length, input_dim).to(device)
            
            model = RegularLSTM(input_dim=input_dim, hidden_units=32)
            model.to(device)
            model.eval()
            
            with torch.no_grad():
                output = model(X)
            
            assert output.shape == (batch_size, 1)
    
    def test_batch_size_consistency(self, device):
        """Test that models handle different batch sizes consistently."""
        input_dim = 25
        sequence_length = 30
        
        model = RegularLSTM(input_dim=input_dim, hidden_units=32)
        model.to(device)
        model.eval()
        
        batch_sizes = [1, 4, 16, 32]
        
        for batch_size in batch_sizes:
            X = torch.randn(batch_size, sequence_length, input_dim).to(device)
            
            with torch.no_grad():
                output = model(X)
            
            assert output.shape == (batch_size, 1)

class TestModelComparison:
    
    def test_model_output_differences(self, sample_sequences, device):
        """Test that different models produce different outputs."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        
        regular_model = RegularLSTM(input_dim=input_dim, hidden_units=32)
        mc_model = MonteCarloLSTM(input_dim=input_dim, hidden_units=32)
        
        regular_model.to(device)
        mc_model.to(device)
        
        regular_model.eval()
        mc_model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        
        with torch.no_grad():
            regular_output = regular_model(X_tensor)
            mc_output = mc_model(X_tensor)
        
        # Models should produce different outputs (with high probability)
        assert not torch.allclose(regular_output, mc_output, atol=0.1)
    
    def test_parameter_count_comparison(self, sample_sequences):
        """Compare parameter counts between models."""
        X, y = sample_sequences
        input_dim = X.shape[2]
        hidden_units = 32
        
        regular_model = RegularLSTM(input_dim=input_dim, hidden_units=hidden_units)
        mc_model = MonteCarloLSTM(input_dim=input_dim, hidden_units=hidden_units)
        
        regular_params = sum(p.numel() for p in regular_model.parameters())
        mc_params = sum(p.numel() for p in mc_model.parameters())
        
        # Monte Carlo model should have additional parameters for attention
        assert mc_params > regular_params
        
        print(f"Regular LSTM parameters: {regular_params}")
        print(f"Monte Carlo LSTM parameters: {mc_params}")
