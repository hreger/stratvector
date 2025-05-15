"""
Interactive Brokers Gateway connector for live trading.
"""

from typing import Dict, List, Optional, Union
import logging
from datetime import datetime
import ssl
from dataclasses import dataclass
from enum import Enum
import threading
import time
from ib_insync import (
    IB, Contract, Order, MarketOrder, LimitOrder, StopOrder,
    Trade, Position, PnLSingle, util
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrderType(str, Enum):
    """Supported order types."""
    MARKET = "MKT"
    LIMIT = "LMT"
    STOP = "STOP"

class OrderStatus(str, Enum):
    """Order status types."""
    PENDING = "Pending"
    SUBMITTED = "Submitted"
    FILLED = "Filled"
    CANCELLED = "Cancelled"
    REJECTED = "Rejected"

@dataclass
class OrderRequest:
    """Order request container."""
    contract: Contract
    order_type: OrderType
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: str = "DAY"

class IBGateway:
    """
    Interactive Brokers Gateway connector.
    
    Parameters
    ----------
    host : str
        Gateway host address
    port : int
        Gateway port number
    client_id : int
        Client ID for connection
    timeout : int
        Connection timeout in seconds
    readonly : bool
        Whether to connect in readonly mode
    use_ssl : bool
        Whether to use SSL/TLS for connection
        
    Attributes
    ----------
    ib : IB
        IB connection instance
    positions : Dict[str, Position]
        Current positions
    orders : Dict[int, Trade]
        Active orders
    pnl : Dict[str, PnLSingle]
        Current PnL data
    """
    
    def __init__(
        self,
        host: str,
        port: int,
        client_id: int,
        timeout: int = 20,
        readonly: bool = False,
        use_ssl: bool = True
    ) -> None:
        self.host = host
        self.port = port
        self.client_id = client_id
        self.timeout = timeout
        self.readonly = readonly
        self.use_ssl = use_ssl
        
        self.ib = IB()
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[int, Trade] = {}
        self.pnl: Dict[str, PnLSingle] = {}
        
        # Connection monitoring
        self._last_heartbeat = time.time()
        self._heartbeat_thread = None
        self._is_connected = False
        
    def connect(self) -> bool:
        """
        Connect to IB Gateway with SSL/TLS.
        
        Returns
        -------
        bool
            True if connection successful
        """
        try:
            # Configure SSL context if needed
            if self.use_ssl:
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = False
                ssl_context.verify_mode = ssl.CERT_NONE
            else:
                ssl_context = None
                
            # Connect to Gateway
            self.ib.connect(
                host=self.host,
                port=self.port,
                clientId=self.client_id,
                timeout=self.timeout,
                readonly=self.readonly,
                ssl=ssl_context
            )
            
            # Set up event handlers
            self.ib.errorEvent += self._on_error
            self.ib.connectedEvent += self._on_connected
            self.ib.disconnectedEvent += self._on_disconnected
            
            # Start heartbeat monitoring
            self._start_heartbeat()
            
            self._is_connected = True
            logger.info("Connected to IB Gateway")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to IB Gateway: {str(e)}")
            return False
            
    def disconnect(self) -> None:
        """Disconnect from IB Gateway."""
        if self._heartbeat_thread:
            self._heartbeat_thread.cancel()
            
        if self.ib.isConnected():
            self.ib.disconnect()
            self._is_connected = False
            logger.info("Disconnected from IB Gateway")
            
    def _start_heartbeat(self) -> None:
        """Start heartbeat monitoring thread."""
        def heartbeat_monitor():
            while self._is_connected:
                if time.time() - self._last_heartbeat > 30:  # 30 second timeout
                    logger.warning("No heartbeat received for 30 seconds")
                    self._handle_connection_drop()
                time.sleep(5)
                
        self._heartbeat_thread = threading.Thread(target=heartbeat_monitor)
        self._heartbeat_thread.daemon = True
        self._heartbeat_thread.start()
        
    def _handle_connection_drop(self) -> None:
        """Handle connection drop."""
        logger.warning("Connection dropped, attempting to reconnect...")
        self.disconnect()
        time.sleep(5)  # Wait before reconnecting
        self.connect()
        
    def _on_error(self, reqId: int, errorCode: int, errorString: str, contract: Optional[Contract]) -> None:
        """Handle error events."""
        if errorCode == 201:  # Order rejected
            logger.error(f"Order rejected: {errorString}")
            if reqId in self.orders:
                self.orders[reqId].orderStatus = OrderStatus.REJECTED
        elif errorCode == 10197:  # Connection lost
            logger.error("Connection lost")
            self._handle_connection_drop()
        else:
            logger.warning(f"IB Error {errorCode}: {errorString}")
            
    def _on_connected(self) -> None:
        """Handle connection events."""
        logger.info("Connected to IB Gateway")
        self._last_heartbeat = time.time()
        
    def _on_disconnected(self) -> None:
        """Handle disconnection events."""
        logger.warning("Disconnected from IB Gateway")
        self._is_connected = False
        
    def place_order(self, request: OrderRequest) -> Optional[Trade]:
        """
        Place an order.
        
        Parameters
        ----------
        request : OrderRequest
            Order request details
            
        Returns
        -------
        Optional[Trade]
            Trade object if order placed successfully
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return None
            
        try:
            # Create order based on type
            if request.order_type == OrderType.MARKET:
                order = MarketOrder(
                    "BUY" if request.quantity > 0 else "SELL",
                    abs(request.quantity),
                    tif=request.time_in_force
                )
            elif request.order_type == OrderType.LIMIT:
                if not request.limit_price:
                    raise ValueError("Limit price required for limit orders")
                order = LimitOrder(
                    "BUY" if request.quantity > 0 else "SELL",
                    abs(request.quantity),
                    request.limit_price,
                    tif=request.time_in_force
                )
            elif request.order_type == OrderType.STOP:
                if not request.stop_price:
                    raise ValueError("Stop price required for stop orders")
                order = StopOrder(
                    "BUY" if request.quantity > 0 else "SELL",
                    abs(request.quantity),
                    request.stop_price,
                    tif=request.time_in_force
                )
            else:
                raise ValueError(f"Unsupported order type: {request.order_type}")
                
            # Place order
            trade = self.ib.placeOrder(request.contract, order)
            self.orders[trade.order.orderId] = trade
            
            logger.info(f"Placed {request.order_type} order for {request.contract.symbol}")
            return trade
            
        except Exception as e:
            logger.error(f"Failed to place order: {str(e)}")
            return None
            
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order.
        
        Parameters
        ----------
        order_id : int
            Order ID to cancel
            
        Returns
        -------
        bool
            True if cancellation successful
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return False
            
        try:
            if order_id in self.orders:
                self.ib.cancelOrder(self.orders[order_id].order)
                logger.info(f"Cancelled order {order_id}")
                return True
            else:
                logger.warning(f"Order {order_id} not found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to cancel order: {str(e)}")
            return False
            
    def get_positions(self) -> Dict[str, Position]:
        """
        Get current positions.
        
        Returns
        -------
        Dict[str, Position]
            Dictionary of positions by symbol
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return {}
            
        try:
            positions = self.ib.positions()
            self.positions = {p.contract.symbol: p for p in positions}
            return self.positions
            
        except Exception as e:
            logger.error(f"Failed to get positions: {str(e)}")
            return {}
            
    def get_pnl(self, account: str = "") -> Dict[str, PnLSingle]:
        """
        Get current PnL data.
        
        Parameters
        ----------
        account : str, optional
            Account ID to get PnL for
            
        Returns
        -------
        Dict[str, PnLSingle]
            Dictionary of PnL data by symbol
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return {}
            
        try:
            pnl = self.ib.pnl(account)
            self.pnl = {p.contract.symbol: p for p in pnl}
            return self.pnl
            
        except Exception as e:
            logger.error(f"Failed to get PnL: {str(e)}")
            return {}
            
    def subscribe_market_data(self, contract: Contract) -> bool:
        """
        Subscribe to market data.
        
        Parameters
        ----------
        contract : Contract
            Contract to subscribe to
            
        Returns
        -------
        bool
            True if subscription successful
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return False
            
        try:
            self.ib.reqMktData(contract)
            logger.info(f"Subscribed to market data for {contract.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to subscribe to market data: {str(e)}")
            return False
            
    def unsubscribe_market_data(self, contract: Contract) -> bool:
        """
        Unsubscribe from market data.
        
        Parameters
        ----------
        contract : Contract
            Contract to unsubscribe from
            
        Returns
        -------
        bool
            True if unsubscription successful
        """
        if not self._is_connected:
            logger.error("Not connected to IB Gateway")
            return False
            
        try:
            self.ib.cancelMktData(contract)
            logger.info(f"Unsubscribed from market data for {contract.symbol}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unsubscribe from market data: {str(e)}")
            return False 