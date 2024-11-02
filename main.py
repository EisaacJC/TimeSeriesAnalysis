import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Tuple
import sys
from visualization.sector_vis import SectorVisualizer
from visualization.plotter import VolatilityPlotter
from models.traditional import TraditionalVolatility
from models.parkinson import ParkinsonVolatility
from models.yang_zhang import YangZhangVolatility
import os
from models.deep_learning import DeepLearningVolatility
from configs.settings import (
    WINDOW_SIZE,
    LOOKBACK_PERIOD,
    TRAIN_SPLIT,
    VAL_SPLIT,
    RANDOM_SEED,
    SECTORS
)
def ensure_results_directory():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)
def load_stock_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        data = data.copy()
        data.loc[:, 'Returns'] = data['Close'].pct_change()
        return data.dropna()
    except Exception as e:
        raise Exception(f"Error loading data for {ticker}: {str(e)}")
def calculate_volatilities(data: pd.DataFrame, window: int = WINDOW_SIZE) -> Tuple[pd.DataFrame, np.ndarray]:
    data = data.copy()
    traditional = TraditionalVolatility(window=window)
    parkinson = ParkinsonVolatility(window=window)
    yang_zhang = YangZhangVolatility(window=window)
    deep_learning = DeepLearningVolatility(window=window)
    data.loc[:, 'Traditional'] = traditional.predict(data['Returns'])
    data.loc[:, 'Parkinson'] = parkinson.predict(data['High'], data['Low'])
    data.loc[:, 'YangZhang'] = yang_zhang.predict(data)
    dl_predictions = deep_learning.predict(data)
    return data, dl_predictions
def analyze_stock(ticker: str) -> Dict:
    logger = logging.getLogger(__name__)
    try:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365 * 2)
        logger.info(f"Loading data for {ticker}")
        data = load_stock_data(ticker, start_date, end_date)
        data, dl_predictions = calculate_volatilities(data)
        total_size = len(data)
        train_size = int(total_size * TRAIN_SPLIT)
        val_size = int(total_size * VAL_SPLIT)
        all_dates = data.index
        train_dates = all_dates[:train_size]
        val_dates = all_dates[train_size:train_size + val_size]
        test_dates = all_dates[train_size + val_size:]
        if isinstance(dl_predictions, np.ndarray):
            test_predictions = pd.Series(
                dl_predictions[train_size + val_size:],
                index=test_dates
            )
        else:
            test_predictions = dl_predictions[train_size + val_size:]

        predictions = {
            'deep_learning': test_predictions
        }
        return {
            'data': data,
            'predictions': predictions,
            'train_dates': train_dates,
            'val_dates': val_dates,
            'test_dates': test_dates
        }
    except Exception as e:
        logger.error(f"Error in analyze_stock for {ticker}: {str(e)}")
        raise

def analyze_sector(sector_name: str) -> Dict:
    logger = logging.getLogger(__name__)
    sector_data = {}
    if sector_name not in SECTORS:
        raise ValueError(f"Invalid sector name: {sector_name}")
    for ticker in SECTORS[sector_name]:
        try:
            logger.info(f"Analyzing {ticker}")
            results = analyze_stock(ticker)
            sector_data[ticker] = results
        except Exception as e:
            logger.error(f"Error analyzing {ticker}: {str(e)}")
            continue
    return sector_data
def display_menu():
    """Display the main menu options."""
    print("\n=== Volatility Analysis Tool ===")
    print("1. Analyze Single Stock")
    print("2. Analyze Sector")
    print("3. Analyze All Sectors")
    print("4. List Available Sectors")
    print("5. List All Stocks")
    print("6. Exit")
    return input("Select an option (1-6): ")
def list_sectors():
    print("\nAvailable Sectors:")
    for i, sector in enumerate(SECTORS.keys(), 1):
        print(f"{i}. {sector}")
def list_stocks():
    print("\nStocks by Sector:")
    for sector, stocks in SECTORS.items():
        print(f"\n{sector}:")
        for stock in stocks:
            print(f"  - {stock}")
def get_sector_choice() -> str:
    list_sectors()
    while True:
        try:
            choice = int(input("\nSelect sector number: "))
            if 1 <= choice <= len(SECTORS):
                return list(SECTORS.keys())[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Please enter a valid number.")
def get_stock_choice() -> str:
    all_stocks = []
    for stocks in SECTORS.values():
        all_stocks.extend(stocks)
    while True:
        ticker = input("\nEnter stock ticker (e.g., AAPL): ").upper()
        if ticker in all_stocks:
            return ticker
        print(f"Invalid ticker. Available stocks: {', '.join(all_stocks)}")


def main():
    logger = setup_logger()
    ensure_results_directory()
    np.random.seed(RANDOM_SEED)
    plotter = VolatilityPlotter()
    sector_viz = SectorVisualizer()

    while True:
        try:
            choice = display_menu()
            if choice == '1':
                ticker = get_stock_choice()
                logger.info(f"Starting analysis for {ticker}")
                results = analyze_stock(ticker)
                fig = plotter.plot_volatility_comparison(
                    data=results['data'],
                    predictions=results['predictions'],
                    train_dates=results['train_dates'],
                    val_dates=results['val_dates'],
                    test_dates=results['test_dates'],
                    title=f"{ticker} Volatility Analysis"
                )
                plotter.save_plot(fig, f'{ticker}_analysis.png')
                logger.info(f"Analysis completed for {ticker}")
            elif choice == '2':
                sector_name = get_sector_choice()
                logger.info(f"Starting analysis for {sector_name} sector")
                sector_data = analyze_sector(sector_name)
                fig = sector_viz.create_sector_dashboard(sector_data, sector_name)
                sector_viz.save_plot(fig, f'{sector_name}_sector_analysis.png')
                logger.info(f"Sector analysis completed for {sector_name}")
            elif choice == '3':
                logger.info("Starting analysis for all sectors")
                for sector_name in SECTORS:
                    logger.info(f"Analyzing {sector_name} sector")
                    sector_data = analyze_sector(sector_name)
                    fig = sector_viz.create_sector_dashboard(sector_data, sector_name)
                    sector_viz.save_plot(fig, f'{sector_name}_sector_analysis.png')
                logger.info("All sector analyses completed")
            elif choice == '4':
                list_sectors()
            elif choice == '5':
                list_stocks()
            elif choice == '6':
                print("Exiting program...")
                break
            else:
                print("Invalid choice. Please try again.")
        except Exception as e:
            logger.error(f"Error in main execution: {str(e)}")
            print(f"An error occurred: {str(e)}")
            continue

if __name__ == "__main__":
    main()
