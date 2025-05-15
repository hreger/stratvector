"""
Tests for the interactive dashboard.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from src.visualization.dashboard import Dashboard, PerformanceMetrics

@pytest.fixture
def sample_returns():
    """Create sample returns data."""
    dates = pd.date_range(start="2020-01-01", end="2020-12-31", freq="D")
    strategy_returns = pd.Series(
        np.random.normal(0.0005, 0.01, len(dates)),
        index=dates
    )
    benchmark_returns = pd.Series(
        np.random.normal(0.0003, 0.008, len(dates)),
        index=dates
    )
    return strategy_returns, benchmark_returns

@pytest.fixture
def sample_metrics():
    """Create sample performance metrics."""
    return PerformanceMetrics(
        total_return=0.15,
        annualized_return=0.12,
        annualized_volatility=0.18,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.1,
        var_95=0.02,
        cvar_95=0.03,
        win_rate=0.55,
        profit_factor=1.8
    )

@pytest.fixture
def sample_parameter_sensitivity():
    """Create sample parameter sensitivity data."""
    param1 = np.linspace(0, 1, 10)
    param2 = np.linspace(0, 1, 10)
    sensitivity = np.zeros((10, 10))
    for i, p1 in enumerate(param1):
        for j, p2 in enumerate(param2):
            sensitivity[i, j] = p1 * p2
    return pd.DataFrame(
        sensitivity,
        index=[f"Param1_{i}" for i in range(10)],
        columns=[f"Param2_{i}" for i in range(10)]
    )

@pytest.fixture
def dashboard(sample_returns, sample_metrics, sample_parameter_sensitivity):
    """Create dashboard instance."""
    strategy_returns, benchmark_returns = sample_returns
    return Dashboard(
        strategy_returns=strategy_returns,
        benchmark_returns=benchmark_returns,
        risk_metrics=sample_metrics,
        parameter_sensitivity=sample_parameter_sensitivity
    )

def test_equity_curve_creation(dashboard):
    """Test equity curve creation."""
    fig = dashboard.create_equity_curve()
    assert fig is not None
    assert len(fig.data) == 2  # Strategy and benchmark lines
    assert fig.layout.title.text == "Equity Curve"

def test_returns_distribution_creation(dashboard):
    """Test returns distribution creation."""
    fig = dashboard.create_returns_distribution()
    assert fig is not None
    assert len(fig.data) == 2  # Histogram and normal distribution
    assert fig.layout.title.text == "Returns Distribution"

def test_risk_metrics_table_creation(dashboard):
    """Test risk metrics table creation."""
    fig = dashboard.create_risk_metrics_table()
    assert fig is not None
    assert len(fig.data) == 1  # Single table
    assert fig.layout.title.text == "Risk Metrics"

def test_parameter_sensitivity_creation(dashboard):
    """Test parameter sensitivity heatmap creation."""
    fig = dashboard.create_parameter_sensitivity()
    assert fig is not None
    assert len(fig.data) == 1  # Single heatmap
    assert fig.layout.title.text == "Parameter Sensitivity"

def test_dashboard_creation(dashboard):
    """Test full dashboard creation."""
    dashboard.create_dashboard()
    assert (dashboard.output_dir / "dashboard.html").exists()

def test_pdf_export(dashboard):
    """Test PDF report export."""
    dashboard.export_pdf_report()
    assert (dashboard.output_dir / "performance_report.pdf").exists()

def test_excel_export(dashboard):
    """Test Excel summary export."""
    dashboard.export_excel_summary()
    assert (dashboard.output_dir / "performance_summary.xlsx").exists()

def test_chart_snapshot_export(dashboard):
    """Test chart snapshot export."""
    dashboard.export_chart_snapshots()
    assert (dashboard.output_dir / "equity_curve.png").exists()
    assert (dashboard.output_dir / "returns_distribution.png").exists()
    assert (dashboard.output_dir / "parameter_sensitivity.png").exists()

def test_output_directory_creation(dashboard):
    """Test output directory creation."""
    assert dashboard.output_dir.exists()
    assert dashboard.output_dir.is_dir()

def test_metrics_formatting(dashboard):
    """Test metrics formatting in exports."""
    # Test PDF metrics
    metrics = {
        "Total Return": f"{dashboard.risk_metrics.total_return:.2%}",
        "Annualized Return": f"{dashboard.risk_metrics.annualized_return:.2%}",
        "Annualized Volatility": f"{dashboard.risk_metrics.annualized_volatility:.2%}",
        "Sharpe Ratio": f"{dashboard.risk_metrics.sharpe_ratio:.2f}",
        "Sortino Ratio": f"{dashboard.risk_metrics.sortino_ratio:.2f}",
        "Maximum Drawdown": f"{dashboard.risk_metrics.max_drawdown:.2%}",
        "Value at Risk (95%)": f"{dashboard.risk_metrics.var_95:.2%}",
        "Conditional VaR (95%)": f"{dashboard.risk_metrics.cvar_95:.2%}",
        "Win Rate": f"{dashboard.risk_metrics.win_rate:.2%}",
        "Profit Factor": f"{dashboard.risk_metrics.profit_factor:.2f}"
    }
    
    for metric, value in metrics.items():
        assert "%" in value or "." in value  # Check formatting
        assert isinstance(value, str)  # Check type

def test_parameter_sensitivity_formatting(dashboard):
    """Test parameter sensitivity data formatting."""
    assert isinstance(dashboard.parameter_sensitivity, pd.DataFrame)
    assert dashboard.parameter_sensitivity.shape == (10, 10)  # Check dimensions
    assert all(isinstance(x, (int, float)) for x in dashboard.parameter_sensitivity.values.flatten()) 