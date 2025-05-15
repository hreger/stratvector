"""
Monte Carlo simulation engine for risk analysis.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from numba import njit, prange, float64, int64
from dataclasses import dataclass
from scipy.stats import norm
import warnings
from concurrent.futures import ThreadPoolExecutor
import time

@dataclass
class RiskMetrics:
    """Container for risk metrics."""
    var_95: float
    cvar_95: float
    max_drawdown: float
    expected_shortfall: float
    worst_path: np.ndarray
    correlation_matrix: np.ndarray
    simulation_time: float
    path_count: int

@njit(float64[:, :, :](float64[:, :], float64[:, :], int64, int64), parallel=True, fastmath=True)
def _simulate_paths_optimized(
    returns: np.ndarray,
    corr_matrix: np.ndarray,
    n_paths: int,
    n_steps: int
) -> np.ndarray:
    """
    Optimized path simulation using Numba.
    
    Parameters
    ----------
    returns : np.ndarray
        Historical returns for each asset
    corr_matrix : np.ndarray
        Correlation matrix between assets
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
        
    Returns
    -------
    np.ndarray
        Simulated paths for each asset
    """
    n_assets = returns.shape[1]
    paths = np.zeros((n_paths, n_steps, n_assets))
    
    # Pre-compute drift and volatility
    drift = np.mean(returns, axis=0)
    vol = np.std(returns, axis=0)
    
    # Pre-compute Cholesky decomposition
    chol = np.linalg.cholesky(corr_matrix)
    
    # Generate all random numbers at once for better cache utilization
    z = np.random.normal(0, 1, (n_paths, n_steps, n_assets))
    
    for i in prange(n_paths):
        # Transform to correlated
        correlated_z = z[i] @ chol.T
        
        # Initialize first step
        paths[i, 0] = 1.0
        
        # Simulate remaining steps
        for t in range(1, n_steps):
            paths[i, t] = paths[i, t-1] * np.exp(
                (drift - 0.5 * vol**2) + vol * correlated_z[t]
            )
            
    return paths

@njit(float64[:, :, :](float64[:, :, :]), parallel=True, fastmath=True)
def _calculate_drawdowns_optimized(paths: np.ndarray) -> np.ndarray:
    """Optimized drawdown calculation using Numba."""
    n_paths, n_steps, n_assets = paths.shape
    drawdowns = np.zeros((n_paths, n_steps, n_assets))
    
    for i in prange(n_paths):
        for j in range(n_assets):
            peak = paths[i, 0, j]
            for t in range(n_steps):
                if paths[i, t, j] > peak:
                    peak = paths[i, t, j]
                drawdowns[i, t, j] = (peak - paths[i, t, j]) / peak
                
    return drawdowns

class MonteCarloEngine:
    """
    Monte Carlo simulation engine for risk analysis.
    
    Parameters
    ----------
    n_paths : int, default=10000
        Number of simulation paths
    n_steps : int, default=252
        Number of time steps (trading days)
    seed : Optional[int], default=None
        Random seed for reproducibility
    n_threads : int, default=None
        Number of threads for parallel computation
        
    Attributes
    ----------
    n_paths : int
        Number of simulation paths
    n_steps : int
        Number of time steps
    seed : Optional[int]
        Random seed
    n_threads : int
        Number of threads for parallel computation
    """
    
    def __init__(
        self,
        n_paths: int = 10000,
        n_steps: int = 252,
        seed: Optional[int] = None,
        n_threads: Optional[int] = None
    ) -> None:
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed
        self.n_threads = n_threads or (cpu_count() - 1)
        
        if seed is not None:
            np.random.seed(seed)
            
    def simulate(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        validate: bool = True
    ) -> RiskMetrics:
        """
        Run Monte Carlo simulation.
        
        Parameters
        ----------
        returns : pd.DataFrame
            Historical returns for each asset
        weights : Optional[np.ndarray]
            Portfolio weights for each asset
        validate : bool, default=True
            Whether to perform overfitting validation
            
        Returns
        -------
        RiskMetrics
            Container with risk metrics
        """
        start_time = time.time()
        
        # Convert returns to numpy array
        returns_array = returns.values
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_array.T)
        
        # Simulate paths using optimized function
        paths = _simulate_paths_optimized(
            returns_array,
            corr_matrix,
            self.n_paths,
            self.n_steps
        )
        
        # Calculate portfolio paths if weights are provided
        if weights is not None:
            portfolio_paths = np.sum(paths * weights, axis=2)
        else:
            portfolio_paths = np.mean(paths, axis=2)
            
        # Calculate drawdowns using optimized function
        drawdowns = _calculate_drawdowns_optimized(paths)
        
        # Calculate final returns for each path
        final_returns = portfolio_paths[:, -1] - 1
        
        # Calculate VaR and CVaR
        var_95 = np.percentile(final_returns, 5)
        cvar_95 = np.mean(final_returns[final_returns <= var_95])
        
        # Find worst path
        worst_path_idx = np.argmin(final_returns)
        worst_path = portfolio_paths[worst_path_idx]
        
        # Calculate maximum drawdown
        max_drawdown = np.max(drawdowns)
        
        # Calculate expected shortfall
        expected_shortfall = np.mean(final_returns[final_returns <= var_95])
        
        simulation_time = time.time() - start_time
        
        # Perform overfitting validation if requested
        if validate:
            self._validate_overfitting(returns_array, final_returns)
        
        return RiskMetrics(
            var_95=var_95,
            cvar_95=cvar_95,
            max_drawdown=max_drawdown,
            expected_shortfall=expected_shortfall,
            worst_path=worst_path,
            correlation_matrix=corr_matrix,
            simulation_time=simulation_time,
            path_count=self.n_paths
        )
        
    def _validate_overfitting(
        self,
        returns: np.ndarray,
        simulated_returns: np.ndarray
    ) -> None:
        """
        Validate simulation results for overfitting.
        
        Parameters
        ----------
        returns : np.ndarray
            Historical returns
        simulated_returns : np.ndarray
            Simulated returns
        """
        # Check if simulated returns distribution matches historical
        historical_mean = np.mean(returns)
        historical_std = np.std(returns)
        simulated_mean = np.mean(simulated_returns)
        simulated_std = np.std(simulated_returns)
        
        # Calculate distribution similarity metrics
        mean_diff = abs(historical_mean - simulated_mean)
        std_diff = abs(historical_std - simulated_std)
        
        # Set thresholds for overfitting detection
        mean_threshold = 0.1 * abs(historical_mean)
        std_threshold = 0.1 * historical_std
        
        if mean_diff > mean_threshold:
            warnings.warn(
                f"Simulated mean ({simulated_mean:.4f}) differs significantly "
                f"from historical mean ({historical_mean:.4f})"
            )
            
        if std_diff > std_threshold:
            warnings.warn(
                f"Simulated std ({simulated_std:.4f}) differs significantly "
                f"from historical std ({historical_std:.4f})"
            )
            
        # Check for unrealistic returns
        max_historical = np.max(np.abs(returns))
        max_simulated = np.max(np.abs(simulated_returns))
        
        if max_simulated > 2 * max_historical:
            warnings.warn(
                f"Simulated returns contain extreme values "
                f"(max: {max_simulated:.4f}, historical max: {max_historical:.4f})"
            )

def risk_engine(
    max_var: float = 0.05,
    max_drawdown: float = 0.2,
    n_paths: int = 10000,
    validate: bool = True
):
    """
    Decorator for risk validation of backtest results.
    
    Parameters
    ----------
    max_var : float, default=0.05
        Maximum allowed Value at Risk
    max_drawdown : float, default=0.2
        Maximum allowed drawdown
    n_paths : int, default=10000
        Number of Monte Carlo simulation paths
    validate : bool, default=True
        Whether to perform overfitting validation
    """
    def decorator(func):
        def wrapper(self, *args, **kwargs):
            # Run the original backtest
            result = func(self, *args, **kwargs)
            
            # Get historical returns
            returns = self.performance['returns']
            
            # Initialize Monte Carlo engine
            mc = MonteCarloEngine(n_paths=n_paths)
            
            # Run simulation
            risk_metrics = mc.simulate(returns.to_frame(), validate=validate)
            
            # Validate risk metrics
            if abs(risk_metrics.var_95) > max_var:
                warnings.warn(
                    f"VaR {abs(risk_metrics.var_95):.2%} exceeds maximum "
                    f"allowed {max_var:.2%}"
                )
                
            if risk_metrics.max_drawdown > max_drawdown:
                warnings.warn(
                    f"Maximum drawdown {risk_metrics.max_drawdown:.2%} exceeds "
                    f"maximum allowed {max_drawdown:.2%}"
                )
                
            # Add risk metrics to result
            if isinstance(result, pd.DataFrame):
                result['var_95'] = risk_metrics.var_95
                result['cvar_95'] = risk_metrics.cvar_95
                result['max_drawdown'] = risk_metrics.max_drawdown
                result['simulation_time'] = risk_metrics.simulation_time
                
            return result
            
        return wrapper
    return decorator 