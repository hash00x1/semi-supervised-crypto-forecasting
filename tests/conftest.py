"""
Pytest configuration and fixtures for DAO Activity Classifier tests
"""

import pytest
import numpy as np
import pandas as pd
import torch
import tempfile
import os
from pathlib import Path

@pytest.fixture
def sample_dao_data():
    """Create sample DAO data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_daos = 5
    
    data = []
    for dao_id in range(n_daos):
        dao_name = f"dao_{dao_id}"
        dates = pd.date_range('2023-01-01', periods=n_samples//n_daos, freq='D')
        
        for i, date in enumerate(dates):
            data.append({
                'Slug_Santiment': dao_name,
                'vote_date': date.strftime('%Y-%m-%d'),
                'marketcap_usd_cleaned': np.random.lognormal(15, 1),
                'price_usd': np.random.lognormal(2, 0.5),
                'price_btc_usd': np.random.lognormal(-2, 0.3),
                'total_votes': np.random.poisson(10),
                'unique_sv_total_1h': np.random.poisson(100),
                'transaction_volume': np.random.lognormal(12, 1),
                'velocity': np.random.lognormal(1, 0.5),
                'dev_activity': np.random.poisson(5),
                'vp_gini': np.random.beta(2, 5),
                'dao_age': i + 1,
                'marketSegment': np.random.choice(['DeFi', 'Gaming', 'Infrastructure']),
                'Total TVL in USD': np.random.lognormal(16, 1),
                'SP500': 4000 + np.random.normal(0, 100),
                'betweenness_monthly': np.random.exponential(0.1),
                'betweenness_quarterly': np.random.exponential(0.05),
                'num_active_proposals': np.random.poisson(3),
                'whale_transaction_count': np.random.poisson(2),
                'net_activity': np.random.poisson(50)
            })
    
    return pd.DataFrame(data)

@pytest.fixture
def sample_processed_data(sample_dao_data):
    """Create sample processed data with encoded features and targets."""
    df = sample_dao_data.copy()
    
    # Add encoded columns
    from sklearn.preprocessing import LabelEncoder
    le_slug = LabelEncoder()
    le_market = LabelEncoder()
    
    df['Slug_Santiment_Encoded'] = le_slug.fit_transform(df['Slug_Santiment'])
    df['marketSegment_Encoded'] = le_market.fit_transform(df['marketSegment'])
    
    # Add target variable
    df['price_trend'] = np.random.choice([0, 1], size=len(df), p=[0.4, 0.6])
    
    # Add some technical indicators
    df['MA_14'] = df.groupby('Slug_Santiment')['marketcap_usd_cleaned'].rolling(14).mean().reset_index(0, drop=True)
    df['EMA_14'] = df.groupby('Slug_Santiment')['marketcap_usd_cleaned'].ewm(span=14).mean().reset_index(0, drop=True)
    df['RSI_14'] = np.random.uniform(20, 80, len(df))
    
    return df

@pytest.fixture
def sample_sequences():
    """Create sample sequence data for LSTM testing."""
    batch_size = 16
    sequence_length = 120
    num_features = 25
    
    X = np.random.randn(batch_size, sequence_length, num_features).astype(np.float32)
    y = np.random.choice([0, 1], size=batch_size).astype(np.float32)
    
    return X, y

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_config(temp_directory):
    """Mock configuration for testing."""
    return {
        'PROJECT_ROOT': temp_directory,
        'DATA_PATH': os.path.join(temp_directory, 'data', 'raw'),
        'PROCESSED_DATA_PATH': os.path.join(temp_directory, 'data', 'processed'),
        'MODEL_PATH': os.path.join(temp_directory, 'models'),
        'RESULTS_PATH': os.path.join(temp_directory, 'results'),
        'WINDOW_SIZE': 30,
        'BATCH_SIZE': 8,
        'LEARNING_RATE': 0.001,
        'NUM_EPOCHS': 5,
        'HIDDEN_UNITS': 32,
        'CONFIDENCE_THRESHOLD': 0.8,
        'VARIANCE_THRESHOLD': 0.05,
        'NUM_MC_SAMPLES': 3,
        'LAG_PERIODS': 3,
        'LAG_FEATURES': ['price_btc_usd', 'Total TVL in USD', 'SP500'],
        'LOG_TRANSFORM_COLUMNS': ['transaction_volume', 'velocity', 'unique_sv_total_1h'],
        'TECHNICAL_INDICATORS': {'MA_WINDOW': 14, 'EMA_WINDOW': 14, 'RSI_WINDOW': 14},
        'GOVERNANCE_WINDOWS': [14, 30],
        'XGBOOST_PARAMS': {
            'max_depth': 3,
            'learning_rate': 0.1,
            'n_estimators': 10,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False
        }
    }

@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device('cpu')  # Use CPU for consistent testing
