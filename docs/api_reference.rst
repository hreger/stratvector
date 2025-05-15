API Reference
============

Core Components
-------------

Strategy
~~~~~~~~

.. autoclass:: stratvector.Strategy
   :members:
   :undoc-members:
   :show-inheritance:

   The base strategy class that all custom strategies should inherit from.

   Example:
   .. code-block:: python

      class CustomStrategy(Strategy):
          def __init__(self, lookback=20):
              self.lookback = lookback
              
          def generate_signals(self, data):
              returns = data['close'].pct_change(self.lookback)
              return np.where(returns > 0, 1, -1)

Backtester
~~~~~~~~~

.. autoclass:: stratvector.Backtester
   :members:
   :undoc-members:
   :show-inheritance:

   The backtesting engine for strategy evaluation.

   Example:
   .. code-block:: python

      backtester = Backtester(
          strategy=CustomStrategy(),
          data_loader=DataLoader(),
          initial_capital=100000
      )
      results = backtester.run()

Live Trading
-----------

IBGateway
~~~~~~~~~

.. autoclass:: stratvector.IBGateway
   :members:
   :undoc-members:
   :show-inheritance:

   Interactive Brokers gateway for live trading.

   Example:
   .. code-block:: python

      gateway = IBGateway(
          host="127.0.0.1",
          port=7496,
          client_id=1
      )
      gateway.connect()

LiveTrader
~~~~~~~~~

.. autoclass:: stratvector.LiveTrader
   :members:
   :undoc-members:
   :show-inheritance:

   Live trading system for strategy execution.

   Example:
   .. code-block:: python

      trader = LiveTrader(
          gateway=gateway,
          strategy=CustomStrategy(),
          risk_manager=RiskManager()
      )
      trader.run()

Risk Management
-------------

RiskManager
~~~~~~~~~~

.. autoclass:: stratvector.RiskManager
   :members:
   :undoc-members:
   :show-inheritance:

   Risk management system for position and portfolio control.

   Example:
   .. code-block:: python

      risk_manager = RiskManager(
          max_position_size=0.15,
          stop_loss=0.05,
          take_profit=0.1
      )

Data Management
-------------

DataLoader
~~~~~~~~~

.. autoclass:: stratvector.DataLoader
   :members:
   :undoc-members:
   :show-inheritance:

   Market data loading and preprocessing.

   Example:
   .. code-block:: python

      loader = DataLoader(
          source="yahoo",
          symbols=["AAPL", "MSFT"]
      )
      data = loader.load_data()

Performance Analysis
-----------------

PerformanceMetrics
~~~~~~~~~~~~~~~~

.. autoclass:: stratvector.PerformanceMetrics
   :members:
   :undoc-members:
   :show-inheritance:

   Performance metrics calculation and analysis.

   Example:
   .. code-block:: python

      metrics = PerformanceMetrics(returns)
      print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
      print(f"Max Drawdown: {metrics.max_drawdown}")

Configuration
-----------

ConfigManager
~~~~~~~~~~~

.. autoclass:: stratvector.ConfigManager
   :members:
   :undoc-members:
   :show-inheritance:

   Configuration management for strategies and systems.

   Example:
   .. code-block:: python

      config = ConfigManager("config/strategy.toml")
      strategy_config = config.get_strategy_config()

Utilities
--------

Data Processing
~~~~~~~~~~~~~

.. autofunction:: stratvector.utils.preprocess_data
.. autofunction:: stratvector.utils.calculate_returns
.. autofunction:: stratvector.utils.calculate_volatility

Example:
.. code-block:: python

   from stratvector.utils import preprocess_data
   
   data = preprocess_data(raw_data, fill_method="ffill")

Risk Calculations
~~~~~~~~~~~~~~~

.. autofunction:: stratvector.utils.calculate_var
.. autofunction:: stratvector.utils.calculate_cvar
.. autofunction:: stratvector.utils.calculate_drawdown

Example:
.. code-block:: python

   from stratvector.utils import calculate_var
   
   var_95 = calculate_var(returns, confidence_level=0.95)

Performance Metrics
~~~~~~~~~~~~~~~~

.. autofunction:: stratvector.utils.calculate_sharpe_ratio
.. autofunction:: stratvector.utils.calculate_sortino_ratio
.. autofunction:: stratvector.utils.calculate_calmar_ratio

Example:
.. code-block:: python

   from stratvector.utils import calculate_sharpe_ratio
   
   sharpe = calculate_sharpe_ratio(returns, risk_free_rate=0.02)

Error Handling
------------

Custom Exceptions
~~~~~~~~~~~~~~~

.. autoexception:: stratvector.exceptions.StrategyError
.. autoexception:: stratvector.exceptions.DataError
.. autoexception:: stratvector.exceptions.OrderError
.. autoexception:: stratvector.exceptions.RiskError

Example:
.. code-block:: python

   from stratvector.exceptions import StrategyError
   
   try:
       strategy.generate_signals(data)
   except StrategyError as e:
       print(f"Strategy error: {e}") 