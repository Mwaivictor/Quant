"""
MT5 Account Synchronization

Integrates RPM with live MT5 account data (balance, equity, positions).
Ensures RPM risk calculations reflect actual account state.

Can use existing MT5ConnectionPool or initialize its own connection for testing.
"""

import logging
import os
from typing import Optional, Dict, List, Tuple, TYPE_CHECKING
from datetime import datetime

if TYPE_CHECKING:
    from arbitrex.raw_layer.mt5_pool import MT5ConnectionPool

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

from .schemas import Position, PortfolioState

LOG = logging.getLogger(__name__)


class MT5AccountSync:
    """
    Synchronizes RPM portfolio state with MT5 account data.
    
    Fetches:
    - Account balance (total capital)
    - Account equity (capital + unrealized PnL)
    - Open positions (symbol, direction, units, entry_price)
    - Realized/unrealized PnL
    
    Preferred: Use existing MT5ConnectionPool (already connected)
    Fallback: Auto-initialize own connection for standalone testing
    """
    
    def __init__(
        self, 
        mt5_pool: Optional['MT5ConnectionPool'] = None,
        auto_initialize: bool = True
    ):
        """
        Initialize MT5 account synchronization.
        
        Args:
            mt5_pool: Existing MT5ConnectionPool (RECOMMENDED - reuses connection)
            auto_initialize: If True and no pool provided, initialize own connection
        """
        self.mt5_available = MT5_AVAILABLE
        self.last_sync_time: Optional[datetime] = None
        self.initialized_by_us = False
        self.mt5_pool = mt5_pool
        
        if not self.mt5_available:
            LOG.warning("MetaTrader5 library not available - MT5 sync disabled")
            return
        
        # If pool provided, use it (preferred)
        if mt5_pool:
            LOG.info("Using existing MT5ConnectionPool for account sync")
            return
        
        # Otherwise, try to initialize own connection if requested
        if auto_initialize and not self.is_mt5_initialized():
            LOG.info("No MT5 pool provided - attempting standalone initialization")
            self._initialize_mt5()
    
        if not self.mt5_available:
            LOG.warning("MetaTrader5 library not available - MT5 sync disabled")
            return
        
        # Try to initialize MT5 if requested and not already initialized
        if auto_initialize and not self.is_mt5_initialized():
            LOG.info("MT5 not initialized - attempting automatic initialization")
            self._initialize_mt5()
    
    def _initialize_mt5(self) -> bool:
        """
        Initialize MT5 connection using environment variables.
        
        NOTE: Only use this for standalone testing!
        In production, pass mt5_pool to reuse existing connection.
        
        Reads from .env:
        - MT5_LOGIN
        - MT5_PASSWORD
        - MT5_SERVER
        - MT5_TERMINAL (optional)
        
        Returns:
            True if initialization successful, False otherwise
        """
        if not self.mt5_available:
            return False
        
        try:
            # Get credentials from environment
            login = os.environ.get('MT5_LOGIN')
            password = os.environ.get('MT5_PASSWORD')
            server = os.environ.get('MT5_SERVER')
            terminal_path = os.environ.get('MT5_TERMINAL')
            
            if not all([login, password, server]):
                LOG.warning(
                    "MT5 credentials not found in environment. "
                    "Set MT5_LOGIN, MT5_PASSWORD, MT5_SERVER in .env"
                )
                return False
            
            # Attempt initialization
            LOG.info(f"Initializing MT5 connection (login={login}, server={server})")
            
            init_params = {
                'login': int(login),
                'password': password,
                'server': server,
            }
            
            if terminal_path:
                init_params['path'] = terminal_path
                ok = mt5.initialize(**init_params)
            else:
                # Try without path (auto-detect)
                ok = mt5.initialize(
                    login=int(login),
                    password=password,
                    server=server
                )
            
            if not ok:
                error = mt5.last_error()
                LOG.error(f"MT5 initialization failed: {error}")
                return False
            
            # Verify connection with account_info
            account_info = mt5.account_info()
            if account_info is None:
                LOG.error("MT5 initialized but account_info is None")
                mt5.shutdown()
                return False
            
            self.initialized_by_us = True
            LOG.info(
                f"✓ MT5 connected successfully (login={account_info.login}, "
                f"server={account_info.server}, balance={account_info.balance:.2f})"
            )
            return True
        
        except Exception as e:
            LOG.exception(f"Failed to initialize MT5: {e}")
            return False
    
    def is_mt5_initialized(self) -> bool:
        """Check if MT5 is initialized and connected"""
        if not self.mt5_available:
            return False
        
        # If using pool, check pool's connection status
        if self.mt5_pool:
            try:
                # Check if any session in pool is connected
                for name, session in self.mt5_pool._sessions.items():
                    if session.status == "CONNECTED" and hasattr(session, 'mt5_initialized') and session.mt5_initialized:
                        return True
                return False
            except Exception as e:
                LOG.debug(f"Error checking MT5 pool status: {e}")
                return False
        
        # Otherwise check direct MT5 connection
        try:
            account_info = mt5.account_info()
            return account_info is not None
        except Exception as e:
            LOG.debug(f"MT5 not initialized: {e}")
            return False
    
    def get_account_info(self) -> Optional[Dict]:
        """
        Get MT5 account information.
        
        Returns:
            Dict with: balance, equity, margin, profit, currency
            None if MT5 not available or not initialized
        """
        if not self.is_mt5_initialized():
            return None
        
        try:
            account = mt5.account_info()
            if account is None:
                LOG.warning("MT5 account_info() returned None")
                return None
            
            return {
                'balance': account.balance,
                'equity': account.equity,
                'margin': account.margin,
                'margin_free': account.margin_free,
                'margin_level': account.margin_level if account.margin > 0 else None,
                'profit': account.profit,
                'currency': account.currency,
                'leverage': account.leverage,
                'login': account.login,
                'server': account.server,
            }
        
        except Exception as e:
            LOG.error(f"Failed to get MT5 account info: {e}")
            return None
    
    def get_open_positions(self) -> List[Dict]:
        """
        Get all open positions from MT5.
        
        Returns:
            List of position dicts with: symbol, type, volume, price_open, time, profit
            Empty list if MT5 not available or no positions
        """
        if not self.is_mt5_initialized():
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None or len(positions) == 0:
                return []
            
            result = []
            for pos in positions:
                result.append({
                    'ticket': pos.ticket,
                    'symbol': pos.symbol,
                    'type': pos.type,  # 0=BUY, 1=SELL
                    'volume': pos.volume,  # Lots
                    'price_open': pos.price_open,
                    'price_current': pos.price_current,
                    'time': datetime.fromtimestamp(pos.time),
                    'profit': pos.profit,
                    'sl': pos.sl,
                    'tp': pos.tp,
                    'comment': pos.comment,
                })
            
            return result
        
        except Exception as e:
            LOG.error(f"Failed to get MT5 positions: {e}")
            return []
    
    def sync_portfolio_state(self, portfolio_state: PortfolioState) -> bool:
        """
        Update portfolio_state with current MT5 account data.
        
        Args:
            portfolio_state: PortfolioState object to update
        
        Returns:
            True if sync successful, False otherwise
        """
        if not self.is_mt5_initialized():
            LOG.debug("MT5 not initialized - skipping sync")
            return False
        
        try:
            # Get account data
            account_info = self.get_account_info()
            if account_info is None:
                LOG.warning("Failed to get account info during sync")
                return False
            
            # Update capital and equity
            portfolio_state.total_capital = account_info['balance']
            portfolio_state.equity = account_info['equity']
            portfolio_state.unrealized_pnl = account_info['profit']
            
            # Update peak equity and drawdown
            if portfolio_state.equity > portfolio_state.peak_equity:
                portfolio_state.peak_equity = portfolio_state.equity
            
            if portfolio_state.peak_equity > 0:
                portfolio_state.current_drawdown = (
                    portfolio_state.peak_equity - portfolio_state.equity
                ) / portfolio_state.peak_equity
            
            # Get open positions
            mt5_positions = self.get_open_positions()
            
            # Clear and rebuild position tracking
            portfolio_state.open_positions.clear()
            portfolio_state.symbol_exposure.clear()
            
            for mt5_pos in mt5_positions:
                # Convert MT5 position to RPM Position
                # MT5: type 0=BUY (long), type 1=SELL (short)
                direction = 1 if mt5_pos['type'] == 0 else -1
                
                # Convert lots to units (1 lot = 100,000 units for FX)
                # Adjust this multiplier based on your broker's contract size
                contract_size = 100000.0  # Standard lot size for FX
                units = mt5_pos['volume'] * contract_size
                
                position = Position(
                    symbol=mt5_pos['symbol'],
                    direction=direction,
                    units=units,
                    entry_price=mt5_pos['price_open'],
                    entry_time=mt5_pos['time'],
                    unrealized_pnl=mt5_pos['profit'],
                )
                
                # Add to portfolio (use ticket as key for uniqueness)
                key = f"{mt5_pos['symbol']}_{mt5_pos['ticket']}"
                portfolio_state.open_positions[key] = position
                
                # Update symbol exposure
                if mt5_pos['symbol'] not in portfolio_state.symbol_exposure:
                    portfolio_state.symbol_exposure[mt5_pos['symbol']] = 0.0
                
                portfolio_state.symbol_exposure[mt5_pos['symbol']] += units * direction
            
            # Update timestamp
            portfolio_state.last_update = datetime.utcnow()
            self.last_sync_time = datetime.utcnow()
            
            LOG.info(
                f"MT5 sync complete: balance=${account_info['balance']:,.2f}, "
                f"equity=${account_info['equity']:,.2f}, "
                f"positions={len(mt5_positions)}"
            )
            
            return True
        
        except Exception as e:
            LOG.exception(f"Failed to sync portfolio state with MT5: {e}")
            return False
    
    def get_sync_stats(self) -> Dict:
        """Get synchronization statistics"""
        return {
            'mt5_available': self.mt5_available,
            'mt5_initialized': self.is_mt5_initialized(),
            'using_mt5_pool': self.mt5_pool is not None,
            'initialized_by_sync': self.initialized_by_us,
            'last_sync_time': self.last_sync_time.isoformat() if self.last_sync_time else None,
        }
    
    def shutdown(self):
        """
        Shutdown MT5 connection if we initialized it.
        
        NOTE: Does NOT shutdown if using mt5_pool (pool manages its own lifecycle).
        """
        if self.initialized_by_us and self.mt5_available and not self.mt5_pool:
            try:
                mt5.shutdown()
                LOG.info("MT5 connection shutdown")
                self.initialized_by_us = False
            except Exception as e:
                LOG.error(f"Error shutting down MT5: {e}")


def create_mt5_synced_portfolio(default_capital: float = 100000.0) -> Tuple[PortfolioState, MT5AccountSync]:
    """
    Create a portfolio state synchronized with MT5 account.
    
    If MT5 is available and initialized, fetches live account data.
    Otherwise, creates portfolio with default_capital.
    
    Args:
        default_capital: Fallback capital if MT5 not available
    
    Returns:
        Tuple of (PortfolioState, MT5AccountSync)
    """
    syncer = MT5AccountSync()
    
    # Try to get MT5 account data
    account_info = syncer.get_account_info()
    
    if account_info:
        # Create portfolio with MT5 data
        LOG.info(f"Creating portfolio from MT5 account (balance=${account_info['balance']:,.2f})")
        portfolio = PortfolioState(
            total_capital=account_info['balance'],
            equity=account_info['equity'],
            peak_equity=account_info['equity'],
        )
        # Sync positions
        syncer.sync_portfolio_state(portfolio)
    else:
        # Create portfolio with default capital
        LOG.info(f"MT5 not available - creating portfolio with default capital ${default_capital:,.2f}")
        portfolio = PortfolioState(total_capital=default_capital)
    
    return portfolio, syncer
