import numpy as np
import pandas as pd
from models.base import BaseVolatilityModel
from configs.settings import WINDOW_SIZE
from typing import Dict, List


class ParkinsonVolatility(BaseVolatilityModel):
    def __init__(self, window: int = 21):
        self.window = window

    def fit(self, data: pd.DataFrame) -> None:
        # Parkinson volatility doesn't need fitting
        pass

    def predict(self, high: pd.Series, low: pd.Series) -> pd.Series:
        """Calculate Parkinson volatility."""
        log_hl = (np.log(high / low)) ** 2
        factor = 1.0 / (4.0 * np.log(2.0))
        return np.sqrt(factor * log_hl.rolling(window=self.window).mean() * 252)

    def evaluate(self, true_values: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        """Evaluate model performance."""
        return {
            'mae': np.mean(np.abs(true_values - predictions)),
            'rmse': np.sqrt(np.mean((true_values - predictions) ** 2)),
            'correlation': np.corrcoef(true_values, predictions)[0, 1]
        }