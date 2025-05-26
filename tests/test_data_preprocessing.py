"""
Tests for data preprocessing utilities
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_preprocessing import (
    log_transform,
    create_lagged_features,
    impute_missing_values,
    create_sequences,
    encode_categorical_features,
    calculate_returns_and_targets,
    calculate_technical_indicators,
    calculate_activity_features,
    prepare_preprocessor
)

class TestDataPreprocessing:
    
    def test_log_transform(self):
        """Test log transformation function."""
        x = np.array([0, 1, 10, 100])
        result = log_transform(x)
        expected = np.log1p(x)
        np.testing.assert_array_almost_equal(result, expected)
    
    def test_encode_categorical_features(self, sample_dao_data):
        """Test categorical feature encoding."""
        df_encoded, encoders = encode_categorical_features(sample_dao_data)
        
        # Check that encoded columns exist
        assert 'Slug_Santiment_Encoded' in df_encoded.columns
        assert 'marketSegment_Encoded' in df_encoded.columns
        
        # Check that encoders are returned
        assert 'Slug_Santiment' in encoders
        assert 'marketSegment' in encoders
        
        # Check that encoded values are integers
        assert df_encoded['Slug_Santiment_Encoded'].dtype in ['int32', 'int64']
        assert df_encoded['marketSegment_Encoded'].dtype in ['int32', 'int64']
        
        # Check that original data is preserved
        assert len(df_encoded) == len(sample_dao_data)
    
    def test_create_lagged_features(self, sample_processed_data):
        """Test lagged feature creation."""
        lag_features = ['price_btc_usd', 'Total TVL in USD']
        lag_periods = 3
        
        result_df = create_lagged_features(sample_processed_data, lag_features, lag_periods)
        
        # Check that lagged columns were created
        for feature in lag_features:
            for lag in range(1, lag_periods + 1):
                assert f'{feature}_lag_{lag}' in result_df.columns
        
        # Check that the number of lagged features is correct
        original_cols = set(sample_processed_data.columns)
        new_cols = set(result_df.columns) - original_cols
        expected_new_cols = len(lag_features) * lag_periods
        assert len(new_cols) == expected_new_cols
    
    def test_calculate_returns_and_targets(self, sample_dao_data):
        """Test return calculation and target creation."""
        result_df = calculate_returns_and_targets(sample_dao_data)
        
        # Check that target variable was created
        assert 'price_trend' in result_df.columns
        assert result_df['price_trend'].dtype in ['int32', 'int64']
        assert set(result_df['price_trend'].unique()).issubset({0, 1})
        
        # Check that return columns were created
        expected_return_cols = ['price_change', 'daily_return', 'btc_daily_return', 
                               'abnormal_return_btc', 'abnormal_return_30days']
        for col in expected_return_cols:
            assert col in result_df.columns
    
    def test_calculate_technical_indicators(self, sample_processed_data):
        """Test technical indicator calculation."""
        windows = {'MA_WINDOW': 14, 'EMA_WINDOW': 14, 'RSI_WINDOW': 14}
        result_df = calculate_technical_indicators(sample_processed_data, windows)
        
        # Check that technical indicators were created
        assert 'MA_14' in result_df.columns
        assert 'EMA_14' in result_df.columns
        assert 'RSI_14' in result_df.columns
        
        # Check RSI is in valid range (allowing for NaN values)
        rsi_values = result_df['RSI_14'].dropna()
        if len(rsi_values) > 0:
            assert rsi_values.min() >= 0
            assert rsi_values.max() <= 100
    
    def test_calculate_activity_features(self, sample_processed_data):
        """Test activity feature calculation."""
        windows = [14, 30]
        result_df = calculate_activity_features(sample_processed_data, windows)
        
        # Check that governance activity features were created
        for window in windows:
            assert f'{window}_day_gov_activity' in result_df.columns
            assert f'{window}_day_ga_ewma' in result_df.columns
            assert f'{window}_social_media_activity' in result_df.columns
            assert f'{window}_day_social_ewma' in result_df.columns
    
    def test_create_sequences(self, sample_processed_data):
        """Test sequence creation for LSTM."""
        window_size = 30
        
        # Ensure we have enough data
        if len(sample_processed_data) < window_size + 1:
            sample_processed_data = pd.concat([sample_processed_data] * 3, ignore_index=True)
        
        X_seq, y_seq, dates = create_sequences(sample_processed_data, window_size)
        
        # Check shapes
        expected_num_sequences = len(sample_processed_data) - window_size
        assert X_seq.shape[0] == expected_num_sequences
        assert X_seq.shape[1] == window_size
        assert len(y_seq) == expected_num_sequences
        assert len(dates) == expected_num_sequences
        
        # Check that sequences are properly formed
        assert X_seq.ndim == 3  # (samples, timesteps, features)
    
    def test_impute_missing_values(self, sample_processed_data):
        """Test missing value imputation."""
        # Introduce some missing values
        df_with_missing = sample_processed_data.copy()
        df_with_missing.loc[10:15, 'marketcap_usd_cleaned'] = np.nan
        df_with_missing.loc[20:25, 'price_usd'] = np.nan
        
        columns_to_impute = ['marketcap_usd_cleaned', 'price_usd']
        result_df = impute_missing_values(df_with_missing, columns_to_impute)
        
        # Check that some missing values were filled
        for col in columns_to_impute:
            original_missing = df_with_missing[col].isna().sum()
            result_missing = result_df[col].isna().sum()
            assert result_missing <= original_missing
    
    def test_prepare_preprocessor(self):
        """Test preprocessor pipeline creation."""
        feature_columns = ['price_usd', 'total_votes', 'Slug_Santiment_Encoded']
        log_transform_columns = ['price_usd']
        
        preprocessor = prepare_preprocessor(feature_columns, log_transform_columns)
        
        # Check that preprocessor is created
        assert preprocessor is not None
        assert hasattr(preprocessor, 'fit_transform')
        
        # Test with sample data
        sample_data = np.random.randn(100, len(feature_columns))
        sample_data[:, 0] = np.abs(sample_data[:, 0])  # Make positive for log transform
        sample_data[:, 2] = np.random.randint(0, 5, 100)  # Categorical data
        
        transformed = preprocessor.fit_transform(sample_data)
        assert transformed.shape[0] == sample_data.shape[0]

class TestDataPreprocessingIntegration:
    
    def test_full_preprocessing_pipeline(self, sample_dao_data, mock_config):
        """Test the full preprocessing pipeline end-to-end."""
        # Step 1: Encode categorical features
        df_encoded, encoders = encode_categorical_features(sample_dao_data)
        
        # Step 2: Calculate returns and targets
        df_with_targets = calculate_returns_and_targets(df_encoded)
        
        # Step 3: Calculate technical indicators
        df_with_tech = calculate_technical_indicators(df_with_targets, mock_config['TECHNICAL_INDICATORS'])
        
        # Step 4: Calculate activity features
        df_with_activity = calculate_activity_features(df_with_tech, mock_config['GOVERNANCE_WINDOWS'])
        
        # Step 5: Create lagged features
        df_final = create_lagged_features(df_with_activity, mock_config['LAG_FEATURES'], mock_config['LAG_PERIODS'])
        
        # Verify final dataset
        assert 'price_trend' in df_final.columns
        assert 'Slug_Santiment_Encoded' in df_final.columns
        assert 'marketSegment_Encoded' in df_final.columns
        
        # Check that we have more features than we started with
        assert df_final.shape[1] > sample_dao_data.shape[1]
        
        # Verify target distribution
        assert set(df_final['price_trend'].dropna().unique()).issubset({0, 1})
