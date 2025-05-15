"""
Tests for risk metrics calculations.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.risk.monte_carlo import MonteCarloEngine
from src.risk.risk_metrics import RiskMetrics

@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    returns = pd.DataFrame(index=dates)
    
    # Generate correlated returns for 3 assets
    np.random.seed(42)
    returns['asset1'] = np.random.normal(0.0005, 0.01, len(dates))
    returns['asset2'] = 0.7 * returns['asset1'] + np.random.normal(0, 0.005, len(dates))
    returns['asset3'] = -0.3 * returns['asset1'] + np.random.normal(0, 0.008, len(dates))
    
    return returns

@pytest.fixture
def monte_carlo_engine(sample_returns):
    """Create Monte Carlo engine instance."""
    return MonteCarloEngine(
        returns=sample_returns,
        n_paths=1000,
        n_steps=252,
        seed=42
    )

class TestRiskMetrics:
    """Test suite for risk metrics calculations."""
    
    def test_monte_carlo_convergence(self, monte_carlo_engine):
        """Test Monte Carlo simulation convergence."""
        # Run with different numbers of paths
        paths = [100, 1000, 10000]
        results = []
        
        for n_paths in paths:
            monte_carlo_engine.n_paths = n_paths
            metrics = monte_carlo_engine.simulate()
            results.append(metrics)
            
        # Test convergence of key metrics
        var_values = [r.var_95 for r in results]
        cvar_values = [r.cvar_95 for r in results]
        
        # Check that results stabilize with more paths
        assert abs(var_values[-1] - var_values[-2]) < abs(var_values[0] - var_values[1])
        assert abs(cvar_values[-1] - cvar_values[-2]) < abs(cvar_values[0] - cvar_values[1])
        
    def test_var_boundary_conditions(self, monte_carlo_engine):
        """Test Value at Risk boundary conditions."""
        # Test with different confidence levels
        confidence_levels = [0.95, 0.99, 0.999]
        var_values = []
        
        for conf in confidence_levels:
            metrics = monte_carlo_engine.simulate(confidence_level=conf)
            var_values.append(metrics.var_95)
            
        # Check that VaR becomes more extreme with higher confidence
        assert var_values[0] > var_values[1] > var_values[2]
        
        # Test with extreme returns
        extreme_returns = pd.DataFrame({
            'asset1': np.random.normal(0, 0.1, 1000)  # Higher volatility
        })
        monte_carlo_engine.returns = extreme_returns
        metrics = monte_carlo_engine.simulate()
        
        # Check that VaR reflects higher risk
        assert abs(metrics.var_95) > abs(monte_carlo_engine.simulate().var_95)
        
    def test_correlation_matrix_validation(self, monte_carlo_engine):
        """Test correlation matrix validation."""
        # Get correlation matrix
        corr_matrix = monte_carlo_engine.correlation_matrix
        
        # Test matrix properties
        assert corr_matrix.shape == (3, 3)  # 3 assets
        assert np.all(np.diag(corr_matrix) == 1.0)  # Diagonal elements
        assert np.all(corr_matrix >= -1) and np.all(corr_matrix <= 1)  # Bounds
        assert np.allclose(corr_matrix, corr_matrix.T)  # Symmetry
        
        # Test positive definiteness
        eigenvalues = np.linalg.eigvals(corr_matrix)
        assert np.all(eigenvalues > 0)
        
        # Test with invalid correlation
        invalid_corr = corr_matrix.copy()
        invalid_corr[0, 1] = 1.5  # Invalid correlation
        
        with pytest.raises(ValueError):
            monte_carlo_engine.correlation_matrix = invalid_corr
            
    def test_drawdown_calculation(self, monte_carlo_engine):
        """Test drawdown calculation."""
        metrics = monte_carlo_engine.simulate()
        
        # Test drawdown properties
        assert metrics.max_drawdown >= 0
        assert metrics.max_drawdown <= 1
        
        # Test worst path properties
        assert len(metrics.worst_path) == monte_carlo_engine.n_steps
        assert metrics.worst_path[0] == 1.0  # Start at 1
        assert metrics.worst_path[-1] <= 1.0  # End below or at 1
        
    def test_risk_metrics_consistency(self, monte_carlo_engine):
        """Test consistency of risk metrics."""
        metrics = monte_carlo_engine.simulate()
        
        # Test metric relationships
        assert metrics.cvar_95 <= metrics.var_95  # CVaR <= VaR
        assert metrics.max_drawdown >= abs(metrics.var_95)  # Max drawdown >= VaR
        
        # Test Sharpe ratio calculation
        if metrics.annualized_volatility > 0:
            assert metrics.sharpe_ratio == metrics.annualized_return / metrics.annualized_volatility
            
    def test_portfolio_risk(self, monte_carlo_engine):
        """Test portfolio risk calculations."""
        # Test with equal weights
        equal_weights = np.array([1/3, 1/3, 1/3])
        metrics_equal = monte_carlo_engine.simulate(weights=equal_weights)
        
        # Test with concentrated weights
        concentrated_weights = np.array([0.8, 0.1, 0.1])
        metrics_concentrated = monte_carlo_engine.simulate(weights=concentrated_weights)
        
        # Concentrated portfolio should have higher risk
        assert metrics_concentrated.annualized_volatility >= metrics_equal.annualized_volatility
        
    def test_stress_testing(self, monte_carlo_engine):
        """Test stress testing scenarios."""
        # Test with market crash scenario
        crash_returns = monte_carlo_engine.returns.copy()
        crash_returns.iloc[-10:] *= -2  # Simulate crash
        monte_carlo_engine.returns = crash_returns
        metrics_crash = monte_carlo_engine.simulate()
        
        # Test with volatility spike
        spike_returns = monte_carlo_engine.returns.copy()
        spike_returns.iloc[-20:] *= 2  # Simulate volatility spike
        monte_carlo_engine.returns = spike_returns
        metrics_spike = monte_carlo_engine.simulate()
        
        # Compare metrics
        assert metrics_crash.max_drawdown > metrics_spike.max_drawdown
        assert metrics_spike.annualized_volatility > metrics_crash.annualized_volatility
        
    def test_risk_decomposition(self, monte_carlo_engine):
        """Test risk decomposition analysis."""
        # Calculate risk contributions
        weights = np.array([0.4, 0.3, 0.3])
        risk_contrib = monte_carlo_engine.calculate_risk_contributions(weights)
        
        # Test risk contribution properties
        assert len(risk_contrib) == len(weights)
        assert np.allclose(sum(risk_contrib), 1.0)  # Sum to 1
        assert np.all(risk_contrib >= 0)  # Non-negative
        
    def test_risk_limits(self, monte_carlo_engine):
        """Test risk limit validation."""
        # Test position limits
        weights = np.array([0.6, 0.5, 0.5])  # Exceeds 100%
        
        with pytest.raises(ValueError):
            monte_carlo_engine.simulate(weights=weights)
            
        # Test concentration limits
        concentrated_weights = np.array([0.8, 0.1, 0.1])
        
        with pytest.raises(ValueError):
            monte_carlo_engine.validate_concentration(concentrated_weights, max_weight=0.5) 