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