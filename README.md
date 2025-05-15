
# StratVector : Automated Trading Backtesting Platform 🚀

[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build Status](https://img.shields.io/github/actions/workflow/status/yourusername/trading-platform/python-package.yml)](https://github.com/yourusername/trading-platform/actions)
[![Code Coverage](https://img.shields.io/codecov/c/github/yourusername/trading-platform)](https://codecov.io/gh/yourusername/trading-platform)

> A production-grade algorithmic trading platform featuring vectorized backtesting, Monte Carlo risk analysis, and Interactive Brokers integration. Built for quantitative researchers and systematic traders.

---

## ✨ Key Features

- 📈 **Vectorized Backtesting Engine** - Pandas-based strategy simulation with nanosecond-level precision
- 🎲 **Risk Management Suite** - Numba-accelerated Monte Carlo simulations with VaR/CVaR metrics
- ⚡ **Live Trading Gateway** - Seamless integration with Interactive Brokers TWS API
- 📊 **Performance Analytics** - Interactive Plotly dashboards with 30+ financial metrics
- 🔄 **Parameter Optimization** - Genetic algorithm-based hyperparameter tuning
- 📁 **Data Pipeline** - Built-in support for Yahoo Finance, Polygon.io, and custom CSV formats

---

## 🚀 Quick Start

### Prerequisites

- Python 3.7+
- Interactive Brokers Gateway (for live trading)
- TA-Lib (recommended for technical indicators)

### Installation

```
# Clone repository
git clone https://github.com/hreger/stratvector
cd stratvector

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install with core dependencies
pip install -e ".[dev]"
```

---

## 📁 Project Structure

```
trading-platform/
├── config/               # Strategy configuration files
├── docs/                 # Documentation & architecture diagrams
├── src/
│   ├── core/            # Backtesting engine and data models
│   ├── risk_management/ # Monte Carlo simulations and risk metrics
│   ├── live_trading/    # IB API integration layer
│   └── visualization/   # Performance dashboards
├── tests/               # Unit and integration tests
└── strategies/          # Example strategy implementations
```

---

## ⚙️ Core Configuration

### Strategy Configuration (strategies/momentum.toml)

```
[parameters]
entry_threshold = 0.02
exit_threshold = -0.01
lookback_period = 14

[risk]
max_drawdown = 0.15
position_size = 0.1
monte_carlo_sims = 1000

[data]
symbols = ["AAPL", "MSFT", "GOOG"]
start_date = "2020-01-01"
end_date = "2023-01-01"
```

---

## 📈 Backtesting Engine

### Vectorized Strategy Example

```
from src.core import VectorizedBacktester

class MomentumStrategy(VectorizedBacktester):
    def calculate_signals(self):
        returns = self.data['close'].pct_change(self.params['lookback_period'])
        self.signals = pd.DataFrame({
            'long': returns > self.params['entry_threshold'],
            'short': returns  float:
    """
    Calculate 95% VaR using Monte Carlo simulation
    
    Parameters:
    returns (np.ndarray): Historical daily returns
    simulations (int): Number of MC simulations
    time_horizon (int): Projection period in days
    
    Returns:
    float: Value at Risk (95% confidence)
    """
    results = np.empty(simulations)
    for i in range(simulations):
        random_returns = np.random.choice(returns, size=time_horizon)
        results[i] = np.prod(1 + random_returns) - 1
    return np.percentile(results, 5)
```

---

## 📊 Performance Analysis

### Key Metrics Calculation

```
def calculate_sharpe(returns: pd.Series,
                    risk_free_rate: float = 0.0) -> float:
    excess_returns = returns - risk_free_rate/252
    return np.sqrt(252) * excess_returns.mean() / excess_returns.std()

def max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding().max()
    trough = equity_curve.expanding().min()
    return (trough - peak).min()
```

---

## 📈 Live Trading Interface

### Interactive Brokers Integration

```
from ib_insync import IB, MarketOrder, LimitOrder

class LiveTradingSession:
    def __init__(self, strategy_config: dict):
        self.ib = IB()
        self.ib.connect('127.0.0.1', 7497, clientId=1)
        
    def execute_order(self, contract, quantity, order_type, limit_price=None):
        if order_type == 'market':
            order = MarketOrder('BUY', quantity)
        else:
            order = LimitOrder('BUY', quantity, limit_price)
        trade = self.ib.placeOrder(contract, order)
        return trade
```

---

## 🛠️ Development Guidelines

### Testing Framework

```
# Run all tests with coverage
pytest tests/ --cov=src --cov-report=html

# Stress test Monte Carlo simulations
python -m tests.stress_test --simulations 10000
```

### Contribution Workflow

1. Fork the repository
2. Create feature branch (`git checkout -b feature/awesome-feature`)
3. Commit changes with semantic messages (`feat: Add neural strategy`)
4. Push to branch (`git push origin feature/awesome-feature`)
5. Open Pull Request with detailed description

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📫 Contact

Project Maintainer: [P Sanjeev Pradeep] - clashersanjeev@gmail.com

Project Link: [https://github.com/hreger/stratvector](https://github.com/hreger/stratvector)

---

## 🙏 Acknowledgments

- [Pandas](https://pandas.pydata.org/) for vectorized operations
- [Numba](https://numba.pydata.org/) for JIT acceleration
- [IB-insync](https://github.com/erdewit/ib_insync) for Interactive Brokers API
- [Plotly](https://plotly.com/) for interactive visualization

---

Made with ❤️ for the quant community by P Sanjeev Pradeep


---
