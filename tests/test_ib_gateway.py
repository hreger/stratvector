"""
Tests for the IB Gateway connector.
"""

import pytest
from unittest.mock import Mock, patch
from ib_insync import Contract, Trade, Position, PnLSingle
from src.live.ib_gateway import (
    IBGateway,
    OrderType,
    OrderStatus,
    OrderRequest
)

@pytest.fixture
def mock_ib():
    """Create mock IB instance."""
    mock = Mock()
    mock.isConnected.return_value = True
    return mock

@pytest.fixture
def gateway(mock_ib):
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

def test_connection(gateway, mock_ib):
    """Test connection handling."""
    # Test successful connection
    mock_ib.connect.return_value = None
    assert gateway.connect()
    mock_ib.connect.assert_called_once()
    
    # Test connection failure
    mock_ib.connect.side_effect = Exception("Connection failed")
    assert not gateway.connect()
    
    # Test disconnection
    gateway.disconnect()
    mock_ib.disconnect.assert_called_once()

def test_place_market_order(gateway, mock_ib):
    """Test placing market order."""
    # Create test contract and order request
    contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
    request = OrderRequest(
        contract=contract,
        order_type=OrderType.MARKET,
        quantity=100
    )
    
    # Mock trade response
    mock_trade = Mock(spec=Trade)
    mock_trade.order.orderId = 1
    mock_ib.placeOrder.return_value = mock_trade
    
    # Place order
    trade = gateway.place_order(request)
    
    assert trade is not None
    assert trade.order.orderId == 1
    assert 1 in gateway.orders
    mock_ib.placeOrder.assert_called_once()

def test_place_limit_order(gateway, mock_ib):
    """Test placing limit order."""
    # Create test contract and order request
    contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
    request = OrderRequest(
        contract=contract,
        order_type=OrderType.LIMIT,
        quantity=100,
        limit_price=150.0
    )
    
    # Mock trade response
    mock_trade = Mock(spec=Trade)
    mock_trade.order.orderId = 1
    mock_ib.placeOrder.return_value = mock_trade
    
    # Place order
    trade = gateway.place_order(request)
    
    assert trade is not None
    assert trade.order.orderId == 1
    assert 1 in gateway.orders
    mock_ib.placeOrder.assert_called_once()

def test_place_stop_order(gateway, mock_ib):
    """Test placing stop order."""
    # Create test contract and order request
    contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
    request = OrderRequest(
        contract=contract,
        order_type=OrderType.STOP,
        quantity=100,
        stop_price=140.0
    )
    
    # Mock trade response
    mock_trade = Mock(spec=Trade)
    mock_trade.order.orderId = 1
    mock_ib.placeOrder.return_value = mock_trade
    
    # Place order
    trade = gateway.place_order(request)
    
    assert trade is not None
    assert trade.order.orderId == 1
    assert 1 in gateway.orders
    mock_ib.placeOrder.assert_called_once()

def test_cancel_order(gateway, mock_ib):
    """Test order cancellation."""
    # Create test order
    mock_trade = Mock(spec=Trade)
    mock_trade.order.orderId = 1
    gateway.orders[1] = mock_trade
    
    # Cancel order
    assert gateway.cancel_order(1)
    mock_ib.cancelOrder.assert_called_once_with(mock_trade.order)
    
    # Test cancelling non-existent order
    assert not gateway.cancel_order(2)

def test_get_positions(gateway, mock_ib):
    """Test getting positions."""
    # Mock positions response
    mock_position = Mock(spec=Position)
    mock_position.contract.symbol = "AAPL"
    mock_ib.positions.return_value = [mock_position]
    
    positions = gateway.get_positions()
    
    assert "AAPL" in positions
    assert positions["AAPL"] == mock_position
    mock_ib.positions.assert_called_once()

def test_get_pnl(gateway, mock_ib):
    """Test getting PnL data."""
    # Mock PnL response
    mock_pnl = Mock(spec=PnLSingle)
    mock_pnl.contract.symbol = "AAPL"
    mock_ib.pnl.return_value = [mock_pnl]
    
    pnl = gateway.get_pnl()
    
    assert "AAPL" in pnl
    assert pnl["AAPL"] == mock_pnl
    mock_ib.pnl.assert_called_once()

def test_market_data_subscription(gateway, mock_ib):
    """Test market data subscription."""
    # Create test contract
    contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
    
    # Test subscription
    assert gateway.subscribe_market_data(contract)
    mock_ib.reqMktData.assert_called_once_with(contract)
    
    # Test unsubscription
    assert gateway.unsubscribe_market_data(contract)
    mock_ib.cancelMktData.assert_called_once_with(contract)

def test_error_handling(gateway):
    """Test error handling."""
    # Test order rejection
    gateway._on_error(1, 201, "Order rejected", None)
    assert gateway.orders[1].orderStatus == OrderStatus.REJECTED
    
    # Test connection lost
    gateway._on_error(1, 10197, "Connection lost", None)
    assert not gateway._is_connected

def test_connection_drop_handling(gateway, mock_ib):
    """Test connection drop handling."""
    # Simulate connection drop
    gateway._handle_connection_drop()
    
    # Verify reconnection attempt
    mock_ib.disconnect.assert_called_once()
    mock_ib.connect.assert_called_once()

def test_invalid_order_requests(gateway):
    """Test invalid order requests."""
    contract = Contract(symbol="AAPL", secType="STK", exchange="SMART", currency="USD")
    
    # Test limit order without price
    request = OrderRequest(
        contract=contract,
        order_type=OrderType.LIMIT,
        quantity=100
    )
    with pytest.raises(ValueError):
        gateway.place_order(request)
    
    # Test stop order without price
    request = OrderRequest(
        contract=contract,
        order_type=OrderType.STOP,
        quantity=100
    )
    with pytest.raises(ValueError):
        gateway.place_order(request) 