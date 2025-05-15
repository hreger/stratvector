"""
Integration tests for live trading functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from src.live.ib_gateway import IBGateway, OrderRequest, OrderType
from src.live.paper_trading import PaperTrading
from src.strategies.base import Strategy
from ib_insync import Contract, Trade, Position

class MockStrategy(Strategy):
    """Mock strategy for testing."""
    def __init__(self, name: str = "MockStrategy"):
        super().__init__(name)
        self.positions = {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate mock signals."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = np.random.choice([-1, 0, 1], size=len(data))
        return signals
        
    def calculate_position_sizes(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Calculate mock position sizes."""
        positions = pd.DataFrame(index=signals.index)
        positions['position'] = signals['signal'] * 0.1  # 10% position size
        return positions

@pytest.fixture
def mock_ib():
    """Create mock IB instance."""
    mock = Mock()
    mock.isConnected.return_value = True
    return mock

@pytest.fixture
def ib_gateway(mock_ib):
    """Create IBGateway instance with mock IB."""
    with patch('src.live.ib_gateway.IB', return_value=mock_ib):
        gateway = IBGateway(
            host="127.0.0.1",
            port=7496,
            client_id=1,
            timeout=20,
            readonly=False,
            use_ssl=True
        )
        gateway.ib = mock_ib
        gateway._is_connected = True
        return gateway

@pytest.fixture
def paper_trading(ib_gateway):
    """Create PaperTrading instance."""
    return PaperTrading(
        gateway=ib_gateway,
        initial_capital=100000,
        commission=0.001
    )

class IntegrationTestLiveTrading:
    """Integration tests for live trading functionality."""
    
    def test_paper_trading_simulation(self, paper_trading, mock_ib):
        """Test paper trading simulation."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Create order request
        request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        # Place order
        trade = paper_trading.place_order(request)
        
        # Verify order execution
        assert trade is not None
        assert trade.order.orderId in paper_trading.orders
        assert paper_trading.positions[contract.symbol] == 100
        
        # Verify position tracking
        positions = paper_trading.get_positions()
        assert contract.symbol in positions
        assert positions[contract.symbol].position == 100
        
        # Verify PnL tracking
        pnl = paper_trading.get_pnl()
        assert contract.symbol in pnl
        
    def test_order_lifecycle(self, paper_trading, mock_ib):
        """Test order lifecycle validation."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Test market order
        market_request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        market_trade = paper_trading.place_order(market_request)
        assert market_trade is not None
        assert market_trade.order.orderId in paper_trading.orders
        
        # Test limit order
        limit_request = OrderRequest(
            contract=contract,
            order_type=OrderType.LIMIT,
            quantity=50,
            limit_price=150.0
        )
        limit_trade = paper_trading.place_order(limit_request)
        assert limit_trade is not None
        assert limit_trade.order.orderId in paper_trading.orders
        
        # Test stop order
        stop_request = OrderRequest(
            contract=contract,
            order_type=OrderType.STOP,
            quantity=25,
            stop_price=140.0
        )
        stop_trade = paper_trading.place_order(stop_request)
        assert stop_trade is not None
        assert stop_trade.order.orderId in paper_trading.orders
        
        # Test order cancellation
        assert paper_trading.cancel_order(limit_trade.order.orderId)
        assert limit_trade.order.orderId not in paper_trading.orders
        
    def test_position_management(self, paper_trading, mock_ib):
        """Test position management."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Test position opening
        request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        trade = paper_trading.place_order(request)
        
        # Verify position
        assert paper_trading.positions[contract.symbol] == 100
        
        # Test position reduction
        reduce_request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=-50
        )
        reduce_trade = paper_trading.place_order(reduce_request)
        
        # Verify updated position
        assert paper_trading.positions[contract.symbol] == 50
        
        # Test position closing
        close_request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=-50
        )
        close_trade = paper_trading.place_order(close_request)
        
        # Verify position closed
        assert contract.symbol not in paper_trading.positions
        
    def test_risk_management(self, paper_trading, mock_ib):
        """Test risk management features."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Test position size limits
        large_request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=1000  # Exceeds position limit
        )
        
        with pytest.raises(ValueError):
            paper_trading.place_order(large_request)
            
        # Test stop loss
        request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        trade = paper_trading.place_order(request)
        
        # Simulate price drop
        paper_trading.update_prices({contract.symbol: 140.0})  # Below stop loss
        
        # Verify stop loss triggered
        assert contract.symbol not in paper_trading.positions
        
    def test_market_data_handling(self, paper_trading, mock_ib):
        """Test market data handling."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Test market data subscription
        assert paper_trading.subscribe_market_data(contract)
        
        # Test price updates
        prices = {contract.symbol: 150.0}
        paper_trading.update_prices(prices)
        
        # Verify price update
        assert paper_trading.prices[contract.symbol] == 150.0
        
        # Test market data unsubscription
        assert paper_trading.unsubscribe_market_data(contract)
        
    def test_error_handling(self, paper_trading, mock_ib):
        """Test error handling."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Test connection loss
        mock_ib.isConnected.return_value = False
        request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        
        with pytest.raises(ConnectionError):
            paper_trading.place_order(request)
            
        # Test order rejection
        mock_ib.isConnected.return_value = True
        mock_ib.placeOrder.side_effect = Exception("Order rejected")
        
        with pytest.raises(Exception):
            paper_trading.place_order(request)
            
    def test_performance_tracking(self, paper_trading, mock_ib):
        """Test performance tracking."""
        # Create test contract
        contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
        
        # Place and execute orders
        request = OrderRequest(
            contract=contract,
            order_type=OrderType.MARKET,
            quantity=100
        )
        trade = paper_trading.place_order(request)
        
        # Update prices
        paper_trading.update_prices({contract.symbol: 155.0})
        
        # Verify PnL calculation
        pnl = paper_trading.get_pnl()
        assert contract.symbol in pnl
        assert pnl[contract.symbol].realizedPnL != 0
        
        # Verify performance metrics
        metrics = paper_trading.calculate_performance_metrics()
        assert 'total_return' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'max_drawdown' in metrics 