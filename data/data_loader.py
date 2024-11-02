import logging
from typing import Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from configs.settings import WINDOW_SIZE, LOOKBACK_PERIOD


class MarketDataLoader:
    """Handle market data loading and preprocessing."""

    def __init__(self, start_date=None, end_date=None):
        self.start_date = start_date
        self.end_date = end_date

    def load_stock_data(self, ticker):
        """Load stock data from Yahoo Finance."""
        try:
            data = yf.download(ticker, start=self.start_date, end=self.end_date)
            return self._preprocess_data(data)
        except Exception as e:
            logger.error(f"Error loading data for {ticker}: {str(e)}")
            raise

    def _preprocess_data(self, data):
        """Preprocess raw market data."""
        data['Returns'] = data['Close'].pct_change()
        return data.dropna()