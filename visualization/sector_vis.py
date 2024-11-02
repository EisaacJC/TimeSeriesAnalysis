import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging
import os
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
def ensure_results_directory():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
class SectorVisualizer:
    def __init__(self, colors: Optional[Dict[str, str]] = None):
        #plt.style.use('seaborn')
        #sns.set_theme(style="whitegrid")
        plt.rcParams['figure.figsize'] = [12, 8]
        plt.rcParams['figure.dpi'] = 100

        self.colors = colors or {
            'Technology': '#2E86C1',
            'Finance': '#28B463',
            'Healthcare': '#E74C3C',
            'Consumer': '#F4D03F',
            'Energy': '#8E44AD'
        }
        self.logger = logging.getLogger(__name__)
    def create_sector_dashboard(self, sector_data: Dict, sector_name: str) -> plt.Figure:
        fig = plt.figure(figsize=(20, 25))
        gs = gridspec.GridSpec(4, 2, height_ratios=[1.2, 1, 1, 1.2])
        ax1 = fig.add_subplot(gs[0, :])
        self._plot_sector_performance(ax1, sector_data)
        ax2 = fig.add_subplot(gs[1, 0])
        self._plot_volatility_heatmap(ax2, sector_data)
        ax3 = fig.add_subplot(gs[1, 1])
        self._plot_risk_return_scatter(ax3, sector_data)
        ax4 = fig.add_subplot(gs[2, 0])
        self._plot_rolling_correlations(ax4, sector_data)
        ax5 = fig.add_subplot(gs[2, 1])
        self._plot_volume_analysis(ax5, sector_data)
        ax6 = fig.add_subplot(gs[3, :])
        self._plot_volatility_distribution(ax6, sector_data)
        plt.suptitle(f'{sector_name} Sector Analysis Dashboard', fontsize=16, y=0.95)
        plt.tight_layout()
        return fig
    def _plot_sector_performance(self, ax, sector_data: Dict):
        for ticker, data in sector_data.items():
            norm_prices = data['data']['Close'] / data['data']['Close'].iloc[0] * 100
            ax.plot(norm_prices.index, norm_prices, label=ticker, alpha=0.7)
        ax.set_title('Relative Price Performance (Base=100)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Price')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    def _plot_volatility_heatmap(self, ax, sector_data: Dict):
        volatilities = {}
        for ticker, data in sector_data.items():
            vol_methods = ['Traditional', 'Parkinson', 'YangZhang']
            vols = [data['data'][method].mean() for method in vol_methods]
            volatilities[ticker] = vols
        df_heatmap = pd.DataFrame(volatilities, index=['Traditional', 'Parkinson', 'YangZhang'])
        sns.heatmap(df_heatmap, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Average Volatility by Method', fontsize=12)
    def _plot_risk_return_scatter(self, ax, sector_data: Dict):
        returns = []
        risks = []
        tickers = []
        for ticker, data in sector_data.items():
            ret = data['data']['Returns'].mean() * 252 * 100  # Annualized return in %
            risk = data['data']['Traditional'].mean() * 100  # Volatility in %
            returns.append(ret)
            risks.append(risk)
            tickers.append(ticker)
        ax.scatter(risks, returns, s=100)
        for i, ticker in enumerate(tickers):
            ax.annotate(ticker, (risks[i], returns[i]), xytext=(5, 5),
                        textcoords='offset points')
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.3)
        ax.axvline(x=0, color='black', linestyle='--', alpha=0.3)
        ax.set_title('Risk-Return Analysis', fontsize=12)
        ax.set_xlabel('Risk (Volatility %)')
        ax.set_ylabel('Expected Return (%)')
    def _plot_rolling_correlations(self, ax, sector_data: Dict):
        returns_dict = {ticker: data['data']['Returns']
                        for ticker, data in sector_data.items()}
        returns_df = pd.DataFrame(returns_dict)
        roll_corr = returns_df.rolling(window=30).corr()
        avg_corr = roll_corr.groupby(level=0).mean().mean(axis=1)
        ax.plot(avg_corr.index, avg_corr, color='blue', linewidth=2)
        ax.set_title('Average Rolling Correlation (30-day)', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
    def _plot_volume_analysis(self, ax, sector_data: Dict):
        for ticker, data in sector_data.items():
            norm_volume = data['data']['Volume'] / data['data']['Volume'].max()
            ax.plot(norm_volume.index, norm_volume, label=ticker, alpha=0.7)
        ax.set_title('Normalized Trading Volume', fontsize=12)
        ax.set_xlabel('Date')
        ax.set_ylabel('Normalized Volume')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_volatility_distribution(self, ax, sector_data: Dict):
        all_vols = []
        labels = []
        for ticker, data in sector_data.items():
            vols = data['data']['Traditional']
            all_vols.extend(vols)
            labels.extend([ticker] * len(vols))
        df_vols = pd.DataFrame({'Volatility': all_vols, 'Stock': labels})
        sns.violinplot(data=df_vols, x='Stock', y='Volatility', ax=ax)
        ax.set_title('Volatility Distribution by Stock', fontsize=12)
        ax.set_xlabel('Stock')
        ax.set_ylabel('Volatility')

    def save_plot(self, fig: plt.Figure, filename: str,
                  dpi: int = 300, bbox_inches: str = 'tight') -> None:
        try:
            ensure_results_directory()
            save_path = os.path.join("results", filename)
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches)
            self.logger.info(f"Plot saved successfully to {save_path}")
        except Exception as e:
            self.logger.error(f"Error saving plot to {save_path}: {str(e)}")
            raise
def analyze_sector(sector_stocks: List[str], start_date: datetime,
                   end_date: datetime) -> Dict:
    from main import analyze_stock
    sector_data = {}
    for ticker in sector_stocks:
        try:
            results = analyze_stock(ticker)
            sector_data[ticker] = results
        except Exception as e:
            logging.error(f"Error analyzing {ticker}: {str(e)}")
            continue

    return sector_data