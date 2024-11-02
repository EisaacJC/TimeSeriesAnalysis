import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import logging
import os
import numpy as np
from utils.logger import setup_logger
import pandas as pd
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from configs.settings import WINDOW_SIZE
def ensure_results_directory():
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
class VolatilityPlotter:
    def __init__(self, colors: Optional[Dict[str, str]] = None):
        self.colors = colors or {
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
        self.logger = logging.getLogger(__name__)
    def plot_volatility_comparison(self, data: pd.DataFrame,
                                   predictions: Dict,
                                   train_dates: pd.DatetimeIndex,
                                   val_dates: pd.DatetimeIndex,
                                   test_dates: pd.DatetimeIndex,
                                   title: str = "Volatility Methods Comparison") -> plt.Figure:
        fig = plt.figure(figsize=(20, 25))
        gs = gridspec.GridSpec(5, 1, height_ratios=[1, 1, 1.2, 1, 1.2])
        ax1 = fig.add_subplot(gs[0])
        self._plot_price_evolution(ax1, data, train_dates, val_dates, test_dates)
        ax2 = fig.add_subplot(gs[1])
        self._plot_returns_distribution(ax2, data, train_dates, val_dates, test_dates)
        ax3 = fig.add_subplot(gs[2])
        self._plot_volatilities(ax3, data, predictions, train_dates, val_dates, test_dates)
        ax4 = fig.add_subplot(gs[3])
        self._plot_error_analysis(ax4, data, predictions, test_dates)
        ax5 = fig.add_subplot(gs[4])
        self._plot_rolling_correlations(ax5, data, predictions, test_dates)
        plt.tight_layout()
        return fig
    def _plot_price_evolution(self, ax: plt.Axes, data: pd.DataFrame,
                              train_dates: pd.DatetimeIndex,
                              val_dates: pd.DatetimeIndex,
                              test_dates: pd.DatetimeIndex) -> None:
        self._add_period_backgrounds(ax, train_dates, val_dates, test_dates)
        ax.plot(data.index, data['Close'], color=self.colors['price'],
                label='Price', linewidth=1.5)
        ax.set_title('Price Evolution', fontsize=14, pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    def _plot_returns_distribution(self, ax: plt.Axes, data: pd.DataFrame,
                                   train_dates: pd.DatetimeIndex,
                                   val_dates: pd.DatetimeIndex,
                                   test_dates: pd.DatetimeIndex) -> None:
        self._add_period_backgrounds(ax, train_dates, val_dates, test_dates)
        ax.plot(data.index, data['Returns'], color=self.colors['returns'],
                label='Returns', alpha=0.7)
        ax.set_title('Returns Distribution', fontsize=14, pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Returns')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    def _plot_volatilities(self, ax: plt.Axes, data: pd.DataFrame,
                           predictions: Dict, train_dates: pd.DatetimeIndex,
                           val_dates: pd.DatetimeIndex,
                           test_dates: pd.DatetimeIndex) -> None:
        self._add_period_backgrounds(ax, train_dates, val_dates, test_dates)
        for method in ['Traditional', 'Parkinson', 'YangZhang', 'GARCH']:
            if method in data.columns:
                ax.plot(data.index, data[method],
                        color=self.colors[method],
                        label=method, linewidth=2 if method == 'Traditional' else 1,
                        alpha=0.7)
        if predictions:
            if 'deep_learning' in predictions:
                ax.plot(test_dates, predictions['deep_learning'],
                        color=self.colors['DeepLearning'],
                        label='Deep Learning', linestyle='--', alpha=0.8)
            if 'gp' in predictions and 'gp_uncertainty' in predictions:
                ax.plot(test_dates, predictions['gp'],
                        color=self.colors['GP'],
                        label='GP', linestyle='--', alpha=0.8)
                ax.fill_between(test_dates,
                                predictions['gp'] - 2 * predictions['gp_uncertainty'],
                                predictions['gp'] + 2 * predictions['gp_uncertainty'],
                                color=self.colors['GP'], alpha=0.2)
        ax.set_title('Volatility Methods Comparison', fontsize=14, pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Volatility')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    def _plot_error_analysis(self, ax: plt.Axes, data: pd.DataFrame,
                             predictions: Dict,
                             test_dates: pd.DatetimeIndex) -> None:
        traditional = data.loc[test_dates, 'Traditional']
        methods = ['Parkinson', 'YangZhang', 'GARCH']
        for method in methods:
            if method in data.columns:
                error = data.loc[test_dates, method] - traditional
                ax.plot(test_dates, error, label=f'{method} Error',
                        alpha=0.7, color=self.colors[method])
        if 'deep_learning' in predictions:
            error = predictions['deep_learning'] - traditional
            ax.plot(test_dates, error, label='Deep Learning Error',
                    alpha=0.7, color=self.colors['DeepLearning'])
        if 'gp' in predictions:
            error = predictions['gp'] - traditional
            ax.plot(test_dates, error, label='GP Error',
                    alpha=0.7, color=self.colors['GP'])
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax.set_title('Error Analysis', fontsize=14, pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Error')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    def _plot_rolling_correlations(self, ax: plt.Axes, data: pd.DataFrame,
                                   predictions: Dict,
                                   test_dates: pd.DatetimeIndex,
                                   window: int = 21) -> None:
        traditional = data.loc[test_dates, 'Traditional']
        methods = ['Parkinson', 'YangZhang', 'GARCH']
        for method in methods:
            if method in data.columns:
                roll_corr = pd.DataFrame({
                    'Traditional': traditional,
                    'Method': data.loc[test_dates, method]
                }).rolling(window=window).corr().unstack().iloc[:, 1]

                ax.plot(test_dates, roll_corr, label=f'{method}',
                        alpha=0.7, color=self.colors[method])
        if 'deep_learning' in predictions:
            roll_corr = pd.DataFrame({
                'Traditional': traditional,
                'Method': predictions['deep_learning']
            }).rolling(window=window).corr().unstack().iloc[:, 1]
            ax.plot(test_dates, roll_corr, label='Deep Learning',
                    alpha=0.7, color=self.colors['DeepLearning'])
        if 'gp' in predictions:
            roll_corr = pd.DataFrame({
                'Traditional': traditional,
                'Method': predictions['gp']
            }).rolling(window=window).corr().unstack().iloc[:, 1]
            ax.plot(test_dates, roll_corr, label='GP',
                    alpha=0.7, color=self.colors['GP'])
        ax.set_title('Rolling Correlations with Traditional Volatility',
                     fontsize=14, pad=20)
        ax.set_xlabel('Date')
        ax.set_ylabel('Correlation')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    def _add_period_backgrounds(self, ax: plt.Axes,
                                train_dates: pd.DatetimeIndex,
                                val_dates: pd.DatetimeIndex,
                                test_dates: pd.DatetimeIndex) -> None:
        ax.axvspan(train_dates[0], train_dates[-1],
                   color=self.colors['training'], alpha=0.2, label='Training')
        ax.axvspan(val_dates[0], val_dates[-1],
                   color=self.colors['validation'], alpha=0.2, label='Validation')
        ax.axvspan(test_dates[0], test_dates[-1],
                   color=self.colors['testing'], alpha=0.2, label='Testing')

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

if __name__ == "__main__":
    import numpy as np

    dates = pd.date_range(start='2020-01-01', end='2022-12-31', freq='B')
    np.random.seed(42)

    data = pd.DataFrame({
        'Close': np.random.randn(len(dates)).cumsum() + 100,
        'Returns': np.random.randn(len(dates)) * 0.02,
        'Traditional': np.abs(np.random.randn(len(dates)) * 0.15 + 0.2),
        'Parkinson': np.abs(np.random.randn(len(dates)) * 0.15 + 0.2),
        'YangZhang': np.abs(np.random.randn(len(dates)) * 0.15 + 0.2),
        'GARCH': np.abs(np.random.randn(len(dates)) * 0.15 + 0.2)
    }, index=dates)

    predictions = {
        'deep_learning': np.abs(np.random.randn(100) * 0.15 + 0.2),
        'gp': np.abs(np.random.randn(100) * 0.15 + 0.2),
        'gp_uncertainty': np.abs(np.random.randn(100) * 0.02)
    }

    train_dates = dates[:400]
    val_dates = dates[400:500]
    test_dates = dates[500:600]

    plotter = VolatilityPlotter()
    fig = plotter.plot_volatility_comparison(data, predictions,
                                             train_dates, val_dates, test_dates)
