"""
Backtesting engine for trading strategies.
"""

from typing import Dict, List, Optional, Union, Protocol
import numpy as np
import pandas as pd
from pathlib import Path
import toml
from abc import ABC, abstractmethod
from ..risk.monte_carlo import risk_engine

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals from market data.
        
        Parameters
        ----------
        data : pd.DataFrame
            Market data with OHLCV columns
            
        Returns
        -------
        pd.DataFrame
            DataFrame with signal columns (1: long, -1: short, 0: neutral)
        """
        pass

class MomentumStrategy(Strategy):
    """Momentum strategy based on price returns."""
    
    def __init__(self, lookback: int = 14, entry_threshold: float = 0.03,
                 exit_threshold: float = -0.02):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate momentum signals."""
        returns = data['close'].pct_change(self.lookback)
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Long entry
        signals.loc[returns > self.entry_threshold, 'signal'] = 1
        # Exit long
        signals.loc[returns < self.exit_threshold, 'signal'] = 0
        
        return signals

class MeanReversionStrategy(Strategy):
    """Mean reversion strategy based on z-scores."""
    
    def __init__(self, lookback: int = 20, z_threshold: float = 2.0):
        self.lookback = lookback
        self.z_threshold = z_threshold
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mean reversion signals."""
        returns = data['close'].pct_change()
        z_scores = (returns - returns.rolling(self.lookback).mean()) / returns.rolling(self.lookback).std()
        
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Short when price is too high
        signals.loc[z_scores > self.z_threshold, 'signal'] = -1
        # Long when price is too low
        signals.loc[z_scores < -self.z_threshold, 'signal'] = 1
        
        return signals

class VectorizedBacktester:
    """
    Vectorized backtesting engine for evaluating trading strategies.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to the strategy configuration file
    data_dir : Union[str, Path]
        Directory containing market data files
        
    Attributes
    ----------
    config : Dict
        Strategy configuration parameters
    positions : pd.DataFrame
        Current positions in the portfolio
    performance : pd.DataFrame
        Historical performance metrics
    """
    
    def __init__(
        self,
        config_path: Union[str, Path],
        data_dir: Union[str, Path],
        strategy: Strategy
    ) -> None:
        self.config = self._load_config(config_path)
        self.data_dir = Path(data_dir)
        self.strategy = strategy
        self.positions = pd.DataFrame()
        self.performance = pd.DataFrame()
        
    def _load_config(self, config_path: Union[str, Path]) -> Dict:
        """Load strategy configuration from TOML file."""
        with open(config_path, 'r') as f:
            return toml.load(f)
            
    def load_data(self, symbol: str) -> pd.DataFrame:
        """
        Load market data for a symbol from CSV or Parquet.
        
        Parameters
        ----------
        symbol : str
            Trading symbol
            
        Returns
        -------
        pd.DataFrame
            Market data with OHLCV columns
        """
        # Try parquet first, then CSV
        parquet_path = self.data_dir / f"{symbol}.parquet"
        csv_path = self.data_dir / f"{symbol}.csv"
        
        if parquet_path.exists():
            return pd.read_parquet(parquet_path)
        elif csv_path.exists():
            return pd.read_csv(csv_path, index_col=0, parse_dates=True)
        else:
            raise FileNotFoundError(f"No data file found for {symbol}")
            
    @risk_engine(max_var=0.05, max_drawdown=0.2)
    def run(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Run the backtest over the specified date range.
        
        Parameters
        ----------
        start_date : str
            Start date in YYYY-MM-DD format
        end_date : str
            End date in YYYY-MM-DD format
            
        Returns
        -------
        pd.DataFrame
            Backtest results including performance metrics
        """
        # Load data for all symbols
        data = {}
        for symbol in self.config['universe']['symbols']:
            data[symbol] = self.load_data(symbol)
            data[symbol] = data[symbol].loc[start_date:end_date]
            
        # Generate signals for each symbol
        signals = {}
        for symbol, symbol_data in data.items():
            signals[symbol] = self.strategy.generate_signals(symbol_data)
            
        # Calculate positions and returns
        self.positions = self._calculate_positions(signals)
        self.performance = self._calculate_performance(data)
        
        return self.performance
        
    def _calculate_positions(self, signals: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate portfolio positions from signals."""
        # Combine signals into a single DataFrame
        positions = pd.DataFrame()
        for symbol, signal_df in signals.items():
            positions[symbol] = signal_df['signal'] * self.config['parameters']['position_size']
            
        # Apply position size limits
        max_size = self.config['parameters']['max_position_size']
        positions = positions.clip(-max_size, max_size)
        
        return positions
        
    def _calculate_performance(self, data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Calculate portfolio performance metrics."""
        # Calculate returns for each position
        returns = pd.DataFrame()
        for symbol in self.positions.columns:
            price_returns = data[symbol]['close'].pct_change()
            position_returns = self.positions[symbol] * price_returns
            returns[symbol] = position_returns
            
        # Calculate portfolio metrics
        portfolio_returns = returns.sum(axis=1)
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        performance = pd.DataFrame()
        performance['returns'] = portfolio_returns
        performance['cumulative_returns'] = cumulative_returns
        performance['drawdown'] = cumulative_returns / cumulative_returns.cummax() - 1
        
        return performance
        
    def calculate_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics for the backtest.
        
        Returns
        -------
        Dict[str, float]
            Dictionary of performance metrics
        """
        if self.performance.empty:
            raise ValueError("Run backtest before calculating metrics")
            
        returns = self.performance['returns']
        cumulative_returns = self.performance['cumulative_returns']
        
        metrics = {
            'total_return': cumulative_returns.iloc[-1] - 1,
            'annualized_return': (1 + returns.mean()) ** 252 - 1,
            'volatility': returns.std() * np.sqrt(252),
            'sharpe_ratio': (returns.mean() / returns.std()) * np.sqrt(252),
            'max_drawdown': self.performance['drawdown'].min(),
            'win_rate': (returns > 0).mean()
        }
        
        return metrics 