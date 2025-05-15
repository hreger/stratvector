"""
Tests for the risk analysis module.
"""

import pytest
import numpy as np
import pandas as pd
from src.risk.monte_carlo import MonteCarloEngine, RiskMetrics
import time

@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.02, len(dates)),
        'MSFT': np.random.normal(0.0001, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0001, 0.02, len(dates))
    }, index=dates)
    return returns

@pytest.fixture
def mc_engine():
    """Create a Monte Carlo engine instance."""
    return MonteCarloEngine(n_paths=1000, n_steps=252, seed=42)

def test_monte_carlo_initialization(mc_engine):
    """Test Monte Carlo engine initialization."""
    assert mc_engine.n_paths == 1000
    assert mc_engine.n_steps == 252
    assert mc_engine.seed == 42

def test_simulation_output(mc_engine, sample_returns):
    """Test simulation output structure."""
    risk_metrics = mc_engine.simulate(sample_returns)
    
    assert isinstance(risk_metrics, RiskMetrics)
    assert isinstance(risk_metrics.var_95, float)
    assert isinstance(risk_metrics.cvar_95, float)
    assert isinstance(risk_metrics.max_drawdown, float)
    assert isinstance(risk_metrics.expected_shortfall, float)
    assert isinstance(risk_metrics.worst_path, np.ndarray)
    assert isinstance(risk_metrics.correlation_matrix, np.ndarray)
    assert isinstance(risk_metrics.simulation_time, float)
    assert isinstance(risk_metrics.path_count, int)

def test_risk_metrics_range(mc_engine, sample_returns):
    """Test that risk metrics are within expected ranges."""
    risk_metrics = mc_engine.simulate(sample_returns)
    
    # VaR should be negative
    assert risk_metrics.var_95 < 0
    # CVaR should be less than or equal to VaR
    assert risk_metrics.cvar_95 <= risk_metrics.var_95
    # Drawdown should be between 0 and 1
    assert 0 <= risk_metrics.max_drawdown <= 1
    # Expected shortfall should be negative
    assert risk_metrics.expected_shortfall < 0

def test_correlation_matrix(mc_engine, sample_returns):
    """Test correlation matrix properties."""
    risk_metrics = mc_engine.simulate(sample_returns)
    corr_matrix = risk_metrics.correlation_matrix
    
    # Check matrix shape
    assert corr_matrix.shape == (3, 3)  # 3 assets
    # Check symmetry
    assert np.allclose(corr_matrix, corr_matrix.T)
    # Check diagonal elements
    assert np.allclose(np.diag(corr_matrix), 1.0)
    # Check correlation bounds
    assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)

def test_worst_path(mc_engine, sample_returns):
    """Test worst path properties."""
    risk_metrics = mc_engine.simulate(sample_returns)
    worst_path = risk_metrics.worst_path
    
    # Check path length
    assert len(worst_path) == mc_engine.n_steps
    # Check path values are positive
    assert np.all(worst_path > 0)
    # Check path is monotonically increasing or decreasing
    assert np.all(np.diff(worst_path) >= 0) or np.all(np.diff(worst_path) <= 0)

def test_portfolio_weights(mc_engine, sample_returns):
    """Test simulation with portfolio weights."""
    weights = np.array([0.4, 0.3, 0.3])
    risk_metrics = mc_engine.simulate(sample_returns, weights)
    
    # Check that metrics are calculated
    assert isinstance(risk_metrics.var_95, float)
    assert isinstance(risk_metrics.cvar_95, float)
    assert isinstance(risk_metrics.max_drawdown, float)

def test_performance_large_simulation():
    """Test performance with large number of paths."""
    # Create engine with 100k paths
    mc = MonteCarloEngine(n_paths=100000, n_steps=252)
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.02, len(dates)),
        'MSFT': np.random.normal(0.0001, 0.02, len(dates)),
        'GOOGL': np.random.normal(0.0001, 0.02, len(dates))
    }, index=dates)
    
    # Run simulation and measure time
    start_time = time.time()
    risk_metrics = mc.simulate(returns)
    simulation_time = time.time() - start_time
    
    # Check performance
    assert simulation_time < 0.1  # Should complete in less than 100ms
    assert risk_metrics.path_count == 100000

def test_overfitting_validation():
    """Test overfitting validation."""
    mc = MonteCarloEngine(n_paths=1000)
    
    # Create historical returns with known distribution
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.02, len(dates))
    }, index=dates)
    
    # Run simulation with validation
    risk_metrics = mc.simulate(returns, validate=True)
    
    # Check that simulation time is recorded
    assert risk_metrics.simulation_time > 0
    
    # Test with unrealistic returns
    unrealistic_returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.1, len(dates))  # Much higher volatility
    }, index=dates)
    
    # Should raise warning about unrealistic returns
    with pytest.warns(UserWarning, match="Simulated returns contain extreme values"):
        mc.simulate(unrealistic_returns, validate=True)

def test_parallel_computation():
    """Test parallel computation performance."""
    # Create engines with different thread counts
    mc_single = MonteCarloEngine(n_paths=100000, n_threads=1)
    mc_multi = MonteCarloEngine(n_paths=100000, n_threads=None)  # Use all available threads
    
    # Create sample data
    dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
    returns = pd.DataFrame({
        'AAPL': np.random.normal(0.0001, 0.02, len(dates))
    }, index=dates)
    
    # Measure performance
    start_single = time.time()
    mc_single.simulate(returns)
    single_time = time.time() - start_single
    
    start_multi = time.time()
    mc_multi.simulate(returns)
    multi_time = time.time() - start_multi
    
    # Multi-threaded should be faster
    assert multi_time < single_time
    # Both should complete in reasonable time
    assert single_time < 0.2
    assert multi_time < 0.1 