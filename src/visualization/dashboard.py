"""
Interactive dashboard for strategy performance visualization.
"""

from typing import Dict, List, Optional, Union, Tuple
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.io as pio
from datetime import datetime
import os
from pathlib import Path
import json
from dataclasses import dataclass
import webbrowser
from jinja2 import Template
import pdfkit
import xlsxwriter

@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    total_return: float
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    win_rate: float
    profit_factor: float

class Dashboard:
    """
    Interactive dashboard for strategy performance visualization.
    
    Parameters
    ----------
    strategy_returns : pd.Series
        Strategy returns series
    benchmark_returns : pd.Series
        Benchmark returns series
    risk_metrics : PerformanceMetrics
        Strategy risk metrics
    parameter_sensitivity : pd.DataFrame
        Parameter sensitivity analysis results
        
    Attributes
    ----------
    strategy_returns : pd.Series
        Strategy returns
    benchmark_returns : pd.Series
        Benchmark returns
    risk_metrics : PerformanceMetrics
        Risk metrics
    parameter_sensitivity : pd.DataFrame
        Parameter sensitivity data
    """
    
    def __init__(
        self,
        strategy_returns: pd.Series,
        benchmark_returns: pd.Series,
        risk_metrics: PerformanceMetrics,
        parameter_sensitivity: pd.DataFrame
    ) -> None:
        self.strategy_returns = strategy_returns
        self.benchmark_returns = benchmark_returns
        self.risk_metrics = risk_metrics
        self.parameter_sensitivity = parameter_sensitivity
        
        # Create output directory if it doesn't exist
        self.output_dir = Path("output")
        self.output_dir.mkdir(exist_ok=True)
        
    def create_equity_curve(self) -> go.Figure:
        """
        Create equity curve plot.
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Calculate cumulative returns
        strategy_cumulative = (1 + self.strategy_returns).cumprod()
        benchmark_cumulative = (1 + self.benchmark_returns).cumprod()
        
        # Create figure
        fig = go.Figure()
        
        # Add strategy line
        fig.add_trace(
            go.Scatter(
                x=strategy_cumulative.index,
                y=strategy_cumulative.values,
                name="Strategy",
                line=dict(color="blue", width=2)
            )
        )
        
        # Add benchmark line
        fig.add_trace(
            go.Scatter(
                x=benchmark_cumulative.index,
                y=benchmark_cumulative.values,
                name="Benchmark",
                line=dict(color="gray", width=2, dash="dash")
            )
        )
        
        # Update layout
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Date",
            yaxis_title="Cumulative Return",
            hovermode="x unified",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
        
    def create_returns_distribution(self) -> go.Figure:
        """
        Create returns distribution plot.
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Add histogram
        fig.add_trace(
            go.Histogram(
                x=self.strategy_returns,
                name="Returns",
                nbinsx=50,
                histnorm="probability",
                marker_color="blue",
                opacity=0.7
            ),
            secondary_y=False
        )
        
        # Add normal distribution
        x = np.linspace(
            self.strategy_returns.min(),
            self.strategy_returns.max(),
            100
        )
        y = np.exp(-(x - self.strategy_returns.mean())**2 / (2 * self.strategy_returns.std()**2))
        y = y / y.sum() * len(self.strategy_returns)
        
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                name="Normal Distribution",
                line=dict(color="red", width=2)
            ),
            secondary_y=True
        )
        
        # Update layout
        fig.update_layout(
            title="Returns Distribution",
            xaxis_title="Return",
            yaxis_title="Frequency",
            showlegend=True,
            template="plotly_white"
        )
        
        return fig
        
    def create_risk_metrics_table(self) -> go.Figure:
        """
        Create risk metrics table.
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Create metrics table
        metrics = {
            "Total Return": f"{self.risk_metrics.total_return:.2%}",
            "Annualized Return": f"{self.risk_metrics.annualized_return:.2%}",
            "Annualized Volatility": f"{self.risk_metrics.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.risk_metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.risk_metrics.sortino_ratio:.2f}",
            "Maximum Drawdown": f"{self.risk_metrics.max_drawdown:.2%}",
            "Value at Risk (95%)": f"{self.risk_metrics.var_95:.2%}",
            "Conditional VaR (95%)": f"{self.risk_metrics.cvar_95:.2%}",
            "Win Rate": f"{self.risk_metrics.win_rate:.2%}",
            "Profit Factor": f"{self.risk_metrics.profit_factor:.2f}"
        }
        
        # Create table
        fig = go.Figure(data=[
            go.Table(
                header=dict(
                    values=["Metric", "Value"],
                    fill_color="lightblue",
                    align="left"
                ),
                cells=dict(
                    values=[list(metrics.keys()), list(metrics.values())],
                    fill_color="white",
                    align="left"
                )
            )
        ])
        
        # Update layout
        fig.update_layout(
            title="Risk Metrics",
            template="plotly_white"
        )
        
        return fig
        
    def create_parameter_sensitivity(self) -> go.Figure:
        """
        Create parameter sensitivity heatmap.
        
        Returns
        -------
        go.Figure
            Plotly figure object
        """
        # Create heatmap
        fig = px.imshow(
            self.parameter_sensitivity,
            color_continuous_scale="RdYlGn",
            aspect="auto"
        )
        
        # Update layout
        fig.update_layout(
            title="Parameter Sensitivity",
            xaxis_title="Parameter 2",
            yaxis_title="Parameter 1",
            template="plotly_white"
        )
        
        return fig
        
    def create_dashboard(self) -> None:
        """Create and display interactive dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Equity Curve",
                "Returns Distribution",
                "Risk Metrics",
                "Parameter Sensitivity"
            ),
            specs=[
                [{"type": "xy"}, {"type": "xy"}],
                [{"type": "table"}, {"type": "heatmap"}]
            ]
        )
        
        # Add equity curve
        equity_curve = self.create_equity_curve()
        for trace in equity_curve.data:
            fig.add_trace(trace, row=1, col=1)
            
        # Add returns distribution
        returns_dist = self.create_returns_distribution()
        for trace in returns_dist.data:
            fig.add_trace(trace, row=1, col=2)
            
        # Add risk metrics table
        risk_table = self.create_risk_metrics_table()
        fig.add_trace(risk_table.data[0], row=2, col=1)
        
        # Add parameter sensitivity
        param_sens = self.create_parameter_sensitivity()
        fig.add_trace(param_sens.data[0], row=2, col=2)
        
        # Update layout
        fig.update_layout(
            height=1000,
            width=1200,
            showlegend=True,
            template="plotly_white"
        )
        
        # Save and display
        fig.write_html(self.output_dir / "dashboard.html")
        webbrowser.open(self.output_dir / "dashboard.html")
        
    def export_pdf_report(self) -> None:
        """Export performance report as PDF."""
        # Create HTML template
        template_str = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Strategy Performance Report</title>
            <style>
                body { font-family: Arial, sans-serif; }
                .section { margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Strategy Performance Report</h1>
            <div class="section">
                <h2>Risk Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
                    {% for metric, value in metrics.items() %}
                    <tr>
                        <td>{{ metric }}</td>
                        <td>{{ value }}</td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
        </body>
        </html>
        """
        
        # Render template
        template = Template(template_str)
        metrics = {
            "Total Return": f"{self.risk_metrics.total_return:.2%}",
            "Annualized Return": f"{self.risk_metrics.annualized_return:.2%}",
            "Annualized Volatility": f"{self.risk_metrics.annualized_volatility:.2%}",
            "Sharpe Ratio": f"{self.risk_metrics.sharpe_ratio:.2f}",
            "Sortino Ratio": f"{self.risk_metrics.sortino_ratio:.2f}",
            "Maximum Drawdown": f"{self.risk_metrics.max_drawdown:.2%}",
            "Value at Risk (95%)": f"{self.risk_metrics.var_95:.2%}",
            "Conditional VaR (95%)": f"{self.risk_metrics.cvar_95:.2%}",
            "Win Rate": f"{self.risk_metrics.win_rate:.2%}",
            "Profit Factor": f"{self.risk_metrics.profit_factor:.2f}"
        }
        html = template.render(metrics=metrics)
        
        # Convert to PDF
        pdf_path = self.output_dir / "performance_report.pdf"
        pdfkit.from_string(html, str(pdf_path))
        
    def export_excel_summary(self) -> None:
        """Export performance summary as Excel."""
        # Create Excel writer
        excel_path = self.output_dir / "performance_summary.xlsx"
        writer = pd.ExcelWriter(excel_path, engine="xlsxwriter")
        
        # Write returns data
        returns_df = pd.DataFrame({
            "Strategy": self.strategy_returns,
            "Benchmark": self.benchmark_returns
        })
        returns_df.to_excel(writer, sheet_name="Returns")
        
        # Write risk metrics
        metrics_df = pd.DataFrame({
            "Metric": [
                "Total Return",
                "Annualized Return",
                "Annualized Volatility",
                "Sharpe Ratio",
                "Sortino Ratio",
                "Maximum Drawdown",
                "Value at Risk (95%)",
                "Conditional VaR (95%)",
                "Win Rate",
                "Profit Factor"
            ],
            "Value": [
                f"{self.risk_metrics.total_return:.2%}",
                f"{self.risk_metrics.annualized_return:.2%}",
                f"{self.risk_metrics.annualized_volatility:.2%}",
                f"{self.risk_metrics.sharpe_ratio:.2f}",
                f"{self.risk_metrics.sortino_ratio:.2f}",
                f"{self.risk_metrics.max_drawdown:.2%}",
                f"{self.risk_metrics.var_95:.2%}",
                f"{self.risk_metrics.cvar_95:.2%}",
                f"{self.risk_metrics.win_rate:.2%}",
                f"{self.risk_metrics.profit_factor:.2f}"
            ]
        })
        metrics_df.to_excel(writer, sheet_name="Risk Metrics", index=False)
        
        # Write parameter sensitivity
        self.parameter_sensitivity.to_excel(writer, sheet_name="Parameter Sensitivity")
        
        # Save Excel file
        writer.close()
        
    def export_chart_snapshots(self) -> None:
        """Export chart snapshots as PNG files."""
        # Export equity curve
        equity_curve = self.create_equity_curve()
        equity_curve.write_image(self.output_dir / "equity_curve.png")
        
        # Export returns distribution
        returns_dist = self.create_returns_distribution()
        returns_dist.write_image(self.output_dir / "returns_distribution.png")
        
        # Export parameter sensitivity
        param_sens = self.create_parameter_sensitivity()
        param_sens.write_image(self.output_dir / "parameter_sensitivity.png") 