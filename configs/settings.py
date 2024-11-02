import os
from datetime import timedelta
from typing import Dict, List
WINDOW_SIZE = 21
LOOKBACK_PERIOD = 21
TRAIN_SPLIT = 0.6
VAL_SPLIT = 0.2
RANDOM_SEED = 42
DEFAULT_ANALYSIS_PERIOD = timedelta(days=365 * 2)
MODEL_PARAMS = {
    'gru_units': [64, 32, 16],
    'dropout_rate': 0.2,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'early_stopping_patience': 10
}
SECTORS = {
    'Technology': ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META'],
    'Finance': ['JPM', 'BAC', 'GS', 'MS', 'BLK'],
    'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'MRK'],
    'Consumer': ['AMZN', 'WMT', 'HD', 'MCD', 'NKE'],
    'Energy': ['XOM', 'CVX', 'COP', 'SLB', 'EOG']
}
PLOT_COLORS = {
    'training': '#90EE90',
    'validation': '#ADD8E6',
    'testing': '#FFCBA4',
    'price': '#1f77b4',
    'returns': '#2ca02c',
    'Traditional': 'black',
    'Parkinson': 'purple',
    'YangZhang': 'blue',
    'GARCH': 'red',
    'DeepLearning': 'orange',
    'GP': 'brown'
}

