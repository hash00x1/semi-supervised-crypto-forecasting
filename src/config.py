"""
Configuration file for DAO Activity Price Movement Prediction
"""

import os

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "results")

# Data Parameters
FEATURE_COLUMNS = ['vote_date','Slug_Santiment_Encoded','marketcap_usd_cleaned', 'vp_gini', 
                   'num_active_proposals', 'total_votes', 'transaction_volume', 'velocity', 
                   'whale_transaction_count', 'unique_sv_total_1h', 'dev_activity', 'dao_age', 
                   'marketSegment_Encoded', 'Total TVL in USD', 'SP500', 'price_btc_usd', 
                   'betweenness_monthly', 'betweenness_quarterly']

LOG_TRANSFORM_COLUMNS = ['transaction_volume', 'SP500', 'velocity', 'unique_sv_total_1h', 
                         'dev_activity', 'price_usd', 'Total TVL in USD']

# Model parameters
WINDOW_SIZE = 120
BATCH_SIZE = 32
LEARNING_RATE = 0.001
PATIENCE = 12
NUM_EPOCHS = 60
HIDDEN_UNITS = 128

# Semi-supervised learning parameters
CONFIDENCE_THRESHOLD = 0.9
VARIANCE_THRESHOLD = 0.05
NUM_MC_SAMPLES = 3
AUGMENTED_EPOCHS = 10
NUM_ITERATIONS = 10

# Technical indicators parameters
TECHNICAL_INDICATORS = {
    'MA_WINDOW': 14,
    'EMA_WINDOW': 14,
    'RSI_WINDOW': 14
}

# Feature engineering parameters
GOVERNANCE_WINDOWS = [14, 30, 60, 90]
SOCIAL_WINDOWS = [14, 30, 60, 90]
LAG_FEATURES = ['price_btc_usd', 'Total TVL in USD', 'SP500', 'transaction_volume', 'unique_sv_total_1h']
LAG_PERIODS = 5

# XGBoost parameters
XGBOOST_PARAMS = {
    'max_depth': 3,
    'learning_rate': 0.05,
    'n_estimators': 100,
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'use_label_encoder': False
}

# Experiment configurations
LABELED_PERCENTAGES = [0.6, 0.8, 1.0]
SEMI_SUPERVISED_METHODS = ['monte_carlo', 'regular']
