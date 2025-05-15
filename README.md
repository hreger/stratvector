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

## Development Types
### Local Development
1. Start the development environment:
   ```bash
   docker-compose up -d
   ```

2. Access the services:
   - Application: http://localhost:8000
   - Grafana: http://localhost:3000 (admin/admin)
   - Prometheus: http://localhost:9091
   - Drift Detection: http://localhost:8080

### OpenStack Deployment

1. Prepare the environment:
   ```bash
   # Set OpenStack credentials
   source openrc.sh
   
   # Create configuration archive
   tar czf config.tar.gz config/
   ```

2. Deploy using Heat:
   ```bash
   openstack stack create -t deploy/openstack/stratvector.hot \
     -e deploy/openstack/parameters.yaml \
     stratvector-stack
   ```

3. Update deployment:
   ```bash
   # Update configuration
   tar czf config.tar.gz config/
   
   # Update stack
   openstack stack update -t deploy/openstack/stratvector.hot \
     -e deploy/openstack/parameters.yaml \
     stratvector-stack
   ```

### Kubernetes Deployment

1. Add required repositories:
   ```bash
   helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
   helm repo add grafana https://grafana.github.io/helm-charts
   helm repo update
   ```

2. Install dependencies:
   ```bash
   helm install prometheus prometheus-community/kube-prometheus-stack
   ```

3. Install StratVector:
   ```bash
   helm install stratvector deploy/kubernetes/stratvector
   ```

4. Upgrade deployment:
   ```bash
   helm upgrade stratvector deploy/kubernetes/stratvector
   ```


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


## Monitoring

### Prometheus Metrics

The application exposes the following metrics:

- `process_resident_memory_bytes`: Memory usage
- `order_latency_seconds`: Order execution latency
- `strategy_drift_score`: Strategy drift detection score

### Alerts

Configured alerts include:

- High memory usage (>1GB for 5 minutes)
- High order latency (>1 second for 1 minute)
- Strategy drift detected (score >0.8 for 5 minutes)

### Grafana Dashboards

Pre-configured dashboards:

1. System Overview
   - Memory usage
   - CPU usage
   - Network I/O

2. Trading Performance
   - Order latency
   - Execution success rate
   - PnL tracking

3. Strategy Monitoring
   - Drift detection scores
   - Signal distribution
   - Position sizes

## Strategy Drift Detection

The drift detection service runs every 6 hours and monitors:

- Feature distribution changes
- Performance degradation
- Market regime shifts

Results are available in Grafana and can trigger alerts.

## Maintenance

### Regular Tasks

1. Daily:
   - Review performance metrics
   - Check system logs
   - Verify market data

2. Weekly:
   - Performance analysis
   - Strategy review
   - Risk assessment

3. Monthly:
   - Strategy optimization
   - Parameter review
   - System upgrade

### Emergency Procedures

1. System Shutdown:
   ```bash
   # Kubernetes
   kubectl scale deployment stratvector --replicas=0
   
   # Docker Compose
   docker-compose down
   ```

2. Data Backup:
   ```bash
   # Backup configuration
   tar czf config_backup.tar.gz config/
   
   # Backup data
   tar czf data_backup.tar.gz data/
   ```

## Contribution Workflow

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
