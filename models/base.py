from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, List

class BaseVolatilityModel(ABC):
    @abstractmethod
    def fit(self, data):
        pass
    @abstractmethod
    def predict(self, data):
        pass
    @abstractmethod
    def evaluate(self, true_values, predictions):
        pass
