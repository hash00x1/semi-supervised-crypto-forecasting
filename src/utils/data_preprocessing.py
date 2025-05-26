"""
Data preprocessing utilities for DAO price movement prediction.
"""

import numpy as np
import pandas as pd
import ta
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from typing import List, Tuple, Dict, Any

def log_transform(x: np.ndarray) -> np.ndarray:
    """Apply log transformation safely."""
    return np.log1p(x)  # np.log1p is log(1 + x), useful for handling zero values

def create_lagged_features(df: pd.DataFrame, lagged_features: List[str], lag: int = 5) -> pd.DataFrame:
    """Create lagged features for time series analysis."""
    df_copy = df.copy()
    for feature in lagged_features:
        for i in range(1, lag + 1):
            df_copy[f'{feature}_lag_{i}'] = df_copy[feature].shift(i)
            # Set lagged features to NaN if the previous day has no corresponding value
            df_copy[f'{feature}_lag_{i}'] = np.where(
                df_copy['Slug_Santiment_Encoded'].shift(i).isna(), 
                np.nan, 
                df_copy[f'{feature}_lag_{i}']
            )
    return df_copy

def impute_missing_values(df: pd.DataFrame, columns: List[str], 
                         group_col: str = 'Slug_Santiment_Encoded') -> pd.DataFrame:
    """Impute missing values by the mean of the previous and subsequent day within each group."""
    df_imputed = df.copy()
    for col in columns:
        df_imputed[col] = df_imputed.groupby(group_col, group_keys=False)[col].apply(
            lambda x: x.fillna((x.shift() + x.shift(-1)) / 2)
        )
    return df_imputed

def transform_per_group(data: pd.DataFrame, group_column: str, transformations: Dict[str, Any]) -> pd.DataFrame:
    """Apply transformations within each group."""
    return data.groupby(group_column).transform(transformations)

def create_sequences(df: pd.DataFrame, window_size: int = 120) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for LSTM training."""
    X_seq, y_seq, dates = [], [], []
    feature_columns = df.columns.drop(['price_trend', 'vote_date'])
    
    for i in range(len(df) - window_size):
        X_seq.append(df[feature_columns].iloc[i:(i + window_size)].to_numpy())
        y_seq.append(df['price_trend'].iloc[i + window_size - 1])
        dates.append(df['vote_date'].iloc[i + window_size - 1])
    
    return np.array(X_seq), np.array(y_seq), np.array(dates)

def encode_categorical_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical features using LabelEncoder."""
    df_encoded = df.copy()
    encoders = {}
    
    # Encode Slug_Santiment
    slug_encoder = LabelEncoder()
    df_encoded['Slug_Santiment_Encoded'] = slug_encoder.fit_transform(df['Slug_Santiment'])
    encoders['Slug_Santiment'] = slug_encoder
    
    # Encode marketSegment
    market_encoder = LabelEncoder()
    df_encoded['marketSegment_Encoded'] = market_encoder.fit_transform(df['marketSegment'])
    encoders['marketSegment'] = market_encoder
    
    return df_encoded, encoders

def calculate_returns_and_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various returns and create target variable."""
    df = df.copy()
    
    # Sort by group and date
    df = df.sort_values(by=['Slug_Santiment', 'vote_date'])
    
    # Calculate previous day prices
    df['price_usd_prev'] = df.groupby('Slug_Santiment')['price_usd'].shift(1)
    df['marketcap_usd_cleaned_prev'] = df.groupby('Slug_Santiment')['marketcap_usd_cleaned'].shift(1)
    df['price_btc_prev'] = df.groupby('Slug_Santiment')['price_btc_usd'].shift(1)
    
    # Replace zeros with NaN
    df.replace({'price_usd_prev': {0: np.nan}, 'marketcap_usd_cleaned_prev': {0: np.nan}, 
                'price_btc_prev': {0: np.nan}, 'price_usd': {0: np.nan}, 
                'marketcap_usd_cleaned': {0: np.nan}, 'price_btc_usd': {0: np.nan}}, inplace=True)
    
    # Calculate returns
    df['price_change'] = np.log(df['price_usd'] / df['price_usd_prev'])
    df['daily_return'] = np.log(df['marketcap_usd_cleaned'] / df['marketcap_usd_cleaned_prev'])
    df['btc_daily_return'] = np.log(df['price_btc_usd'] / df['price_btc_prev'])
    
    # Clean infinite values
    for col in ['price_change', 'daily_return', 'btc_daily_return']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Calculate historical return
    df['historical_daily_return'] = df['daily_return'].rolling(window=30).mean()
    
    # Calculate abnormal returns
    df['abnormal_return_btc'] = np.log(
        (df['marketcap_usd_cleaned'] / df['marketcap_usd_cleaned_prev']) - 
        (df['price_btc_usd'] / df['price_btc_prev'])
    )
    df['abnormal_return_30days'] = np.log(
        (df['marketcap_usd_cleaned'] / df['marketcap_usd_cleaned_prev']) - 
        df['historical_daily_return']
    )
    
    # Clean abnormal returns
    for col in ['abnormal_return_btc', 'abnormal_return_30days']:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Create target variable
    df['price_trend'] = (df['abnormal_return_30days'] > 0).astype(int)
    
    # Clean up temporary columns
    df.drop(columns=['price_usd_prev', 'marketcap_usd_cleaned_prev', 'price_btc_prev'], 
            inplace=True, errors='ignore')
    
    return df

def calculate_technical_indicators(df: pd.DataFrame, windows: Dict[str, int]) -> pd.DataFrame:
    """Calculate technical indicators."""
    df = df.copy()
    df['vote_date'] = pd.to_datetime(df['vote_date'])
    
    # Moving Average
    df['MA_14'] = df['marketcap_usd_cleaned'].rolling(window=windows['MA_WINDOW']).mean()
    
    # Exponential Moving Average
    df['EMA_14'] = df['marketcap_usd_cleaned'].ewm(span=windows['EMA_WINDOW'], adjust=False).mean()
    
    # RSI
    df['RSI_14'] = ta.momentum.RSIIndicator(
        df['marketcap_usd_cleaned'], window=windows['RSI_WINDOW']
    ).rsi()
    
    return df

def calculate_activity_features(df: pd.DataFrame, windows: List[int]) -> pd.DataFrame:
    """Calculate governance, social media, and network activity features."""
    df = df.copy()
    
    # Governance activity features
    for window in windows:
        # Moving averages
        df[f'{window}_day_gov_activity'] = df['total_votes'].rolling(window=window).sum()
        df[f'{window}_day_ga_ewma'] = df['total_votes'].ewm(span=window, adjust=False).mean()
        
        # Social media activity
        df[f'{window}_social_media_activity'] = df['unique_sv_total_1h'].rolling(window=window).sum()
        df[f'{window}_day_social_ewma'] = df['unique_sv_total_1h'].ewm(span=window, adjust=False).mean()
        
        # Network activity
        df[f'{window}_network_activity'] = df['net_activity'].rolling(window=window).sum()
        df[f'{window}_day_network_ewma'] = df['net_activity'].ewm(span=window, adjust=False).mean()
    
    # Lagged variables
    for window in windows:
        if window >= 14:
            df[f'gov_activity_lag_{window}'] = df[f'{window}_day_gov_activity'].shift(window)
            df[f'social_media_activity_lag_{window}'] = df[f'{window}_social_media_activity'].shift(window)
            df[f'network_activity_lag_{window}'] = df[f'{window}_network_activity'].shift(window)
    
    return df

def prepare_preprocessor(feature_columns: List[str], log_transform_columns: List[str]) -> ColumnTransformer:
    """Prepare the preprocessing pipeline."""
    preprocessor = ColumnTransformer(
        transformers=[
            ('log', Pipeline([
                ('log_transform', FunctionTransformer(log_transform)),
                ('scaler', StandardScaler())
            ]), log_transform_columns),
            ('num', StandardScaler(), [col for col in feature_columns 
                                      if col not in log_transform_columns + 
                                      ['Slug_Santiment_Encoded', 'marketSegment_Encoded', 'dao_age']]),
            ('cat', 'passthrough', ['Slug_Santiment_Encoded', 'marketSegment_Encoded', 'dao_age'])
        ]
    )
    return preprocessor