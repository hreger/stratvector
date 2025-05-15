Welcome to StratVector's documentation!
====================================

StratVector is a comprehensive algorithmic trading framework that enables strategy development, backtesting, and live trading.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   architecture
   strategy_guide
   api_reference
   live_trading
   examples/index

Quick Start
----------

Installation
~~~~~~~~~~~

.. code-block:: bash

   pip install stratvector

Basic Usage
~~~~~~~~~~

.. code-block:: python

   from stratvector import Strategy, Backtester
   
   # Create a simple momentum strategy
   class MomentumStrategy(Strategy):
       def generate_signals(self, data):
           return data.pct_change(periods=20).shift(1)
   
   # Run backtest
   backtester = Backtester(strategy=MomentumStrategy())
   results = backtester.run()

Features
--------

* Strategy Development
  * Custom strategy creation
  * Parameter optimization
  * Risk management
  * Performance analysis

* Backtesting
  * Vectorized backtesting engine
  * Realistic market simulation
  * Transaction costs
  * Slippage modeling

* Live Trading
  * Interactive Brokers integration
  * Paper trading
  * Risk monitoring
  * Performance tracking

* Data Management
  * Market data integration
  * Data preprocessing
  * Feature engineering
  * Portfolio analytics

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 