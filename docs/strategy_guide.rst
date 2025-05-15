Strategy Development Guide
========================

Creating Custom Strategies
------------------------

1. Basic Strategy Structure
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from stratvector import Strategy
   import pandas as pd
   import numpy as np
   
   class CustomStrategy(Strategy):
       def __init__(self, lookback=20, threshold=0.02):
           self.lookback = lookback
           self.threshold = threshold
           
       def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
           """Generate trading signals from market data."""
           signals = pd.DataFrame(index=data.index)
           
           # Calculate momentum
           returns = data['close'].pct_change(periods=self.lookback)
           
           # Generate signals
           signals['signal'] = np.where(returns > self.threshold, 1,
                                      np.where(returns < -self.threshold, -1, 0))
           
           return signals
           
       def calculate_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
           """Calculate position sizes based on signals."""
           positions = pd.DataFrame(index=signals.index)
           positions['position'] = signals['signal'] * 0.1  # 10% position size
           return positions

2. Advanced Strategy Features
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class AdvancedStrategy(Strategy):
       def __init__(self, lookback=20, threshold=0.02, volatility_lookback=60):
           self.lookback = lookback
           self.threshold = threshold
           self.volatility_lookback = volatility_lookback
           
       def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
           signals = pd.DataFrame(index=data.index)
           
           # Calculate returns and volatility
           returns = data['close'].pct_change()
           volatility = returns.rolling(self.volatility_lookback).std()
           
           # Calculate momentum with volatility adjustment
           momentum = returns.rolling(self.lookback).mean() / volatility
           
           # Generate signals with dynamic threshold
           signals['signal'] = np.where(momentum > self.threshold, 1,
                                      np.where(momentum < -self.threshold, -1, 0))
           
           return signals
           
       def calculate_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
           positions = pd.DataFrame(index=signals.index)
           
           # Dynamic position sizing based on volatility
           volatility = signals['signal'].rolling(20).std()
           base_size = 0.1  # 10% base position size
           
           positions['position'] = signals['signal'] * base_size / (1 + volatility)
           return positions

Parameter Optimization
--------------------

1. Grid Search
~~~~~~~~~~~~

.. code-block:: python

   from stratvector import Optimizer
   
   # Define parameter grid
   param_grid = {
       'lookback': [10, 20, 30, 40],
       'threshold': [0.01, 0.02, 0.03],
       'volatility_lookback': [30, 60, 90]
   }
   
   # Create optimizer
   optimizer = Optimizer(
       strategy_class=AdvancedStrategy,
       param_grid=param_grid,
       metric='sharpe_ratio'
   )
   
   # Run optimization
   results = optimizer.optimize(
       data=data,
       start_date='2020-01-01',
       end_date='2023-12-31'
   )

2. Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from stratvector import BayesianOptimizer
   
   # Define parameter space
   param_space = {
       'lookback': (10, 50),
       'threshold': (0.01, 0.05),
       'volatility_lookback': (30, 120)
   }
   
   # Create optimizer
   optimizer = BayesianOptimizer(
       strategy_class=AdvancedStrategy,
       param_space=param_space,
       metric='sharpe_ratio',
       n_trials=50
   )
   
   # Run optimization
   results = optimizer.optimize(
       data=data,
       start_date='2020-01-01',
       end_date='2023-12-31'
   )

Risk Management
--------------

1. Position Sizing
~~~~~~~~~~~~~~~~

.. code-block:: python

   class RiskAwareStrategy(Strategy):
       def __init__(self, max_position_size=0.15, max_drawdown=0.1):
           self.max_position_size = max_position_size
           self.max_drawdown = max_drawdown
           
       def calculate_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
           positions = pd.DataFrame(index=signals.index)
           
           # Calculate base position size
           base_size = signals['signal'] * 0.1
           
           # Apply position limits
           positions['position'] = np.clip(
               base_size,
               -self.max_position_size,
               self.max_position_size
           )
           
           return positions

2. Stop Loss and Take Profit
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   class StopLossStrategy(Strategy):
       def __init__(self, stop_loss=0.05, take_profit=0.1):
           self.stop_loss = stop_loss
           self.take_profit = take_profit
           
       def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
           signals = pd.DataFrame(index=data.index)
           
           # Calculate returns
           returns = data['close'].pct_change()
           
           # Generate signals
           signals['signal'] = np.where(returns > 0, 1, -1)
           
           # Apply stop loss
           cumulative_returns = (1 + returns).cumprod()
           drawdown = cumulative_returns / cumulative_returns.cummax() - 1
           
           signals.loc[drawdown < -self.stop_loss, 'signal'] = 0
           
           # Apply take profit
           signals.loc[returns > self.take_profit, 'signal'] = 0
           
           return signals

Best Practices
-------------

1. Strategy Development
~~~~~~~~~~~~~~~~~~~~

- Start with simple strategies and gradually add complexity
- Use proper data preprocessing and feature engineering
- Implement robust error handling
- Document strategy logic and assumptions
- Test with different market conditions

2. Risk Management
~~~~~~~~~~~~~~~~

- Implement position size limits
- Use stop losses and take profits
- Monitor drawdowns
- Diversify across assets and strategies
- Regular performance review

3. Performance Analysis
~~~~~~~~~~~~~~~~~~~~

- Use multiple performance metrics
- Compare against benchmarks
- Analyze drawdowns and recovery
- Monitor transaction costs
- Regular strategy review

4. Code Organization
~~~~~~~~~~~~~~~~~

- Follow PEP 8 style guide
- Use type hints
- Write comprehensive tests
- Document code and APIs
- Use version control 