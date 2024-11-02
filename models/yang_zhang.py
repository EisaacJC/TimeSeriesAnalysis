import numpy as np
import pandas as pd
from models.base import BaseVolatilityModel
from configs.settings import WINDOW_SIZE
from typing import Dict, List
class YangZhangVolatility(BaseVolatilityModel):
    def __init__(self, window: int = 21):
        self.window = window

    def fit(self, data: pd.DataFrame) -> None:
        pass

    def predict(self, data: pd.DataFrame) -> pd.Series:
        log_co = np.log(data['Open'] / data['Close'].shift(1))
        vo = log_co.rolling(window=self.window).var()
        log_oc = np.log(data['Close'] / data['Open'])
        vc = log_oc.rolling(window=self.window).var()
        log_ho = np.log(data['High'] / data['Open'])
        log_lo = np.log(data['Low'] / data['Open'])
        log_hc = np.log(data['High'] / data['Close'])
        log_lc = np.log(data['Low'] / data['Close'])
        rs = log_ho * log_hc + log_lo * log_lc
        vrs = rs.rolling(window=self.window).mean()
        k = 0.34 / (1.34 + (self.window + 1) / (self.window - 1))
        return np.sqrt((vo + k * vc + (1 - k) * vrs) * 252)

    def evaluate(self, true_values: pd.Series, predictions: pd.Series) -> Dict[str, float]:
        return {
            'mae': np.mean(np.abs(true_values - predictions)),
            'rmse': np.sqrt(np.mean((true_values - predictions) ** 2)),
            'correlation': np.corrcoef(true_values, predictions)[0, 1]
        }
