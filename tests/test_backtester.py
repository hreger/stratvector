"""
Tests for the backtester module.
"""

import pytest
from pathlib import Path
import pandas as pd
import numpy as np
from src.core.backtester import (
    VectorizedBacktester,
    MomentumStrategy,
    MeanReversionStrategy
)
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.backtest.backtester import Backtester
from src.strategies.base import Strategy
from src.data.data_loader import DataLoader

@pytest.fixture
def config_path(tmp_path):
    """Create a temporary config file for testing."""
    config = {
        "parameters": {
            "lookback_period": 14,
            "holding_period": 5,
            "rebalance_frequency": "1d",
            "position_size": 0.1,
            "entry_threshold": 0.03,
            "exit_threshold": -0.02,
            "max_position_size": 0.1
        },
        "universe": {
            "symbols": ["AAPL", "MSFT"],
            "exchange": "SMART",
            "currency": "USD"
        }
    }
    config_file = tmp_path / "test_strategy.toml"
    with open(config_file, "w") as f:
        f.write(str(config))
    return config_file

@pytest.fixture
def data_dir(tmp_path):
    """Create sample market data for testing."""
    # Create sample data for AAPL
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    data = pd.DataFrame({
        'open': np.random.normal(150, 5, len(dates)),
        'high': np.random.normal(155, 5, len(dates)),
        'low': np.random.normal(145, 5, len(dates)),
        'close': np.random.normal(150, 5, len(dates)),
        'volume': np.random.normal(1000000, 100000, len(dates))
    }, index=dates)
    
    # Save as CSV
    data.to_csv(tmp_path / "AAPL.csv")
    
    # Create similar data for MSFT
    data = pd.DataFrame({
        'open': np.random.normal(250, 10, len(dates)),
        'high': np.random.normal(255, 10, len(dates)),
        'low': np.random.normal(245, 10, len(dates)),
        'close': np.random.normal(250, 10, len(dates)),
        'volume': np.random.normal(2000000, 200000, len(dates))
    }, index=dates)
    
    data.to_csv(tmp_path / "MSFT.csv")
    return tmp_path

@pytest.fixture
def momentum_strategy():
    """Create a momentum strategy instance."""
    return MomentumStrategy(lookback=14, entry_threshold=0.03, exit_threshold=-0.02)

@pytest.fixture
def mean_reversion_strategy():
    """Create a mean reversion strategy instance."""
    return MeanReversionStrategy(lookback=20, z_threshold=2.0)

@pytest.fixture
def backtester(config_path, data_dir, momentum_strategy):
    """Create a VectorizedBacktester instance for testing."""
    return VectorizedBacktester(config_path, data_dir, momentum_strategy)

def test_backtester_initialization(backtester):
    """Test that the backtester initializes correctly."""
    assert isinstance(backtester.positions, pd.DataFrame)
    assert isinstance(backtester.performance, pd.DataFrame)
    assert "parameters" in backtester.config
    assert "universe" in backtester.config

def test_load_config(config_path):
    """Test that configuration loading works correctly."""
    backtester = VectorizedBacktester(config_path, Path(), MomentumStrategy())
    assert backtester.config["parameters"]["lookback_period"] == 14
    assert backtester.config["universe"]["symbols"] == ["AAPL", "MSFT"]

def test_momentum_strategy_signals(momentum_strategy):
    """Test momentum strategy signal generation."""
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    data = pd.DataFrame({
        'close': np.linspace(100, 110, len(dates))  # Upward trend
    }, index=dates)
    
    signals = momentum_strategy.generate_signals(data)
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert signals['signal'].isin([-1, 0, 1]).all()

def test_mean_reversion_strategy_signals(mean_reversion_strategy):
    """Test mean reversion strategy signal generation."""
    # Create sample data with mean reversion
    dates = pd.date_range(start='2023-01-01', end='2023-01-31', freq='D')
    data = pd.DataFrame({
        'close': np.sin(np.linspace(0, 4*np.pi, len(dates))) * 10 + 100
    }, index=dates)
    
    signals = mean_reversion_strategy.generate_signals(data)
    assert isinstance(signals, pd.DataFrame)
    assert 'signal' in signals.columns
    assert signals['signal'].isin([-1, 0, 1]).all()

def test_backtest_run(backtester):
    """Test running a backtest."""
    performance = backtester.run('2023-01-01', '2023-12-31')
    assert isinstance(performance, pd.DataFrame)
    assert 'returns' in performance.columns
    assert 'cumulative_returns' in performance.columns
    assert 'drawdown' in performance.columns

def test_performance_metrics(backtester):
    """Test performance metrics calculation."""
    backtester.run('2023-01-01', '2023-12-31')
    metrics = backtester.calculate_metrics()
    
    assert isinstance(metrics, dict)
    assert 'total_return' in metrics
    assert 'annualized_return' in metrics
    assert 'volatility' in metrics
    assert 'sharpe_ratio' in metrics
    assert 'max_drawdown' in metrics
    assert 'win_rate' in metrics

class MockStrategy(Strategy):
    """Mock strategy for testing."""
    def __init__(self, name: str = "MockStrategy"):
        super().__init__(name)
        self.positions = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock signals."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.random.choice([-1, 0, 1], size=len(data))
        return signals
        
    def calculate_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock position sizes."""
        positions = pd.DataFrame(index=signals.index)
        positions['position'] = signals['signal'] * 0.1  # 10% position size
        return positions

@pytest.fixture
def mock_data():
    """Create mock market data."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    data = pd.DataFrame(index=dates)
    
    # Generate random price data
    data['open'] = np.random.normal(100, 1, len(dates))
    data['high'] = data['open'] + np.random.uniform(0, 2, len(dates))
    data['low'] = data['open'] - np.random.uniform(0, 2, len(dates))
    data['close'] = (data['high'] + data['low']) / 2
    data['volume'] = np.random.randint(1000, 10000, len(dates))
    
    return data

@pytest.fixture
def mock_benchmark():
    """Create mock benchmark data."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    returns = pd.Series(
        np.random.normal(0.0003, 0.01, len(dates)),
        index=dates
    )
    return returns

@pytest.fixture
def backtester(mock_data, mock_benchmark):
    """Create backtester instance with mock data."""
    strategy = MockStrategy()
    data_loader = Mock(spec=DataLoader)
    data_loader.load_data.return_value = mock_data
    
    return Backtester(
        strategy=strategy,
        data_loader=data_loader,
        benchmark_returns=mock_benchmark,
        initial_capital=100000,
        commission=0.001
    )

class TestBacktester:
    """Test suite for the backtester."""
    
    def test_initialization(self, backtester):
        """Test backtester initialization."""
        assert backtester.initial_capital == 100000
        assert backtester.commission == 0.001
        assert isinstance(backtester.strategy, MockStrategy)
        
    def test_run_backtest(self, backtester, mock_data):
        """Test running backtest."""
        results = backtester.run()
        
        assert isinstance(results, dict)
        assert 'returns' in results
        assert 'positions' in results
        assert 'trades' in results
        assert len(results['returns']) == len(mock_data)
        
    def test_strategy_validation(self, backtester):
        """Test strategy validation."""
        # Test invalid strategy
        with pytest.raises(TypeError):
            Backtester(
                strategy="invalid",
                data_loader=Mock(spec=DataLoader),
                benchmark_returns=pd.Series(),
                initial_capital=100000
            )
            
        # Test strategy without required methods
        class InvalidStrategy:
            pass
            
        with pytest.raises(AttributeError):
            Backtester(
                strategy=InvalidStrategy(),
                data_loader=Mock(spec=DataLoader),
                benchmark_returns=pd.Series(),
                initial_capital=100000
            )
            
    def test_benchmark_comparison(self, backtester):
        """Test benchmark comparison."""
        results = backtester.run()
        
        # Test benchmark alignment
        assert len(results['returns']) == len(backtester.benchmark_returns)
        assert results['returns'].index.equals(backtester.benchmark_returns.index)
        
        # Test performance comparison
        strategy_cumulative = (1 + results['returns']).cumprod()
        benchmark_cumulative = (1 + backtester.benchmark_returns).cumprod()
        
        assert isinstance(strategy_cumulative, pd.Series)
        assert isinstance(benchmark_cumulative, pd.Series)
        
    def test_position_sizing(self, backtester):
        """Test position sizing logic."""
        results = backtester.run()
        
        # Test position limits
        assert results['positions'].abs().max() <= 1.0  # Max position size 100%
        
        # Test position changes
        position_changes = results['positions'].diff().abs()
        assert position_changes.max() <= 0.2  # Max position change 20%
        
    def test_commission_impact(self, backtester):
        """Test commission impact on returns."""
        # Run with commission
        results_with_commission = backtester.run()
        
        # Run without commission
        backtester.commission = 0
        results_without_commission = backtester.run()
        
        # Compare returns
        assert results_with_commission['returns'].sum() < results_without_commission['returns'].sum()
        
    def test_risk_metrics(self, backtester):
        """Test risk metrics calculation."""
        results = backtester.run()
        
        assert 'sharpe_ratio' in results
        assert 'max_drawdown' in results
        assert 'volatility' in results
        
        # Test metric ranges
        assert results['sharpe_ratio'] >= -10 and results['sharpe_ratio'] <= 10
        assert results['max_drawdown'] >= 0 and results['max_drawdown'] <= 1
        assert results['volatility'] >= 0
        
    def test_trade_execution(self, backtester):
        """Test trade execution logic."""
        results = backtester.run()
        
        # Test trade generation
        assert len(results['trades']) > 0
        
        # Test trade properties
        for trade in results['trades']:
            assert 'entry_date' in trade
            assert 'exit_date' in trade
            assert 'entry_price' in trade
            assert 'exit_price' in trade
            assert 'position' in trade
            assert 'pnl' in trade
            
    def test_data_validation(self, backtester, mock_data):
        """Test data validation."""
        # Test missing data
        invalid_data = mock_data.copy()
        invalid_data.loc[invalid_data.index[0], 'close'] = np.nan
        
        with pytest.raises(ValueError):
            backtester.data_loader.load_data.return_value = invalid_data
            backtester.run()
            
        # Test data frequency
        invalid_data = mock_data.copy()
        invalid_data.index = pd.date_range(
            start="2020-01-01",
            end="2020-12-31",
            freq="W"  # Weekly instead of daily
        )
        
        with pytest.raises(ValueError):
            backtester.data_loader.load_data.return_value = invalid_data
            backtester.run() 