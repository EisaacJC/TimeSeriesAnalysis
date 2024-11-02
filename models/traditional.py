import numpy as np
import pandas as pd
from models.base import BaseVolatilityModel
from configs.settings import WINDOW_SIZE
from typing import Dict, List
class TraditionalVolatility(BaseVolatilityModel):
    def __init__(self, window=21):
        self.window = window
    def fit(self, data):
        pass
    def predict(self, returns):
        return returns.rolling(window=self.window).std() * np.sqrt(252)
    def evaluate(self, true_values, predictions):
        return {
            'mae': np.mean(np.abs(true_values - predictions)),
            'rmse': np.sqrt(np.mean((true_values - predictions) ** 2)),
            'correlation': np.corrcoef(true_values, predictions)[0, 1]
        }
