"""
State Persistence Module

Handles saving and loading of portfolio state to prevent data loss on crashes/restarts.
Supports multiple backends: Redis (primary), JSON file (backup).
"""

import json
import os
from typing import Optional
from datetime import datetime
from pathlib import Path

from .schemas import PortfolioState


class StateManager:
    """
    Manages persistent storage of portfolio state.
    
    Provides automatic save on updates and recovery on startup.
    Supports Redis for distributed systems and JSON files for simplicity.
    """
    
    def __init__(
        self,
        storage_type: str = "file",
        redis_url: Optional[str] = None,
        state_file_path: Optional[str] = None,
    ):
        """
        Initialize state manager.
        
        Args:
            storage_type: "redis" or "file"
            redis_url: Redis connection URL (for redis type)
            state_file_path: Path to JSON state file (for file type)
        """
        self.storage_type = storage_type
        self.redis_client = None
        
        if storage_type == "redis":
            try:
                import redis
                self.redis_client = redis.from_url(redis_url or "redis://localhost:6379/0")
                self.redis_key = "rpm:portfolio_state"
            except ImportError:
                raise ImportError("redis package required for Redis storage. Install: pip install redis")
        
        elif storage_type == "file":
            if state_file_path is None:
                # Default to logs/rpm_state.json
                state_file_path = "logs/rpm_state.json"
            
            self.state_file_path = Path(state_file_path)
            # Create directory if doesn't exist
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        else:
            raise ValueError(f"Invalid storage_type: {storage_type}. Must be 'redis' or 'file'")
    
    def save_state(self, portfolio_state: PortfolioState) -> bool:
        """
        Save portfolio state to persistent storage.
        
        Args:
            portfolio_state: Current portfolio state
        
        Returns:
            bool: True if save successful, False otherwise
        """
        try:
            state_dict = portfolio_state.to_dict()
            state_dict['last_saved'] = datetime.now().isoformat()
            
            if self.storage_type == "redis":
                # Save to Redis with 24 hour TTL
                self.redis_client.setex(
                    self.redis_key,
                    86400,  # 24 hours
                    json.dumps(state_dict)
                )
            
            elif self.storage_type == "file":
                # Save to JSON file with atomic write
                temp_path = self.state_file_path.with_suffix('.tmp')
                with open(temp_path, 'w') as f:
                    json.dump(state_dict, f, indent=2)
                
                # Atomic rename
                temp_path.replace(self.state_file_path)
            
            return True
        
        except Exception as e:
            # Log error but don't crash
            print(f"ERROR saving portfolio state: {e}")
            return False
    
    def load_state(self, default_capital: float = 100000.0) -> PortfolioState:
        """
        Load portfolio state from persistent storage.
        
        Args:
            default_capital: Default capital if no state found
        
        Returns:
            PortfolioState: Loaded state or fresh state if none exists
        """
        try:
            state_dict = None
            
            if self.storage_type == "redis":
                data = self.redis_client.get(self.redis_key)
                if data:
                    state_dict = json.loads(data)
            
            elif self.storage_type == "file":
                if self.state_file_path.exists():
                    with open(self.state_file_path, 'r') as f:
                        state_dict = json.load(f)
            
            if state_dict:
                # Reconstruct PortfolioState from dict
                portfolio_state = PortfolioState.from_dict(state_dict)
                print(f"✓ Portfolio state loaded from {self.storage_type}")
                print(f"  - Open positions: {len(portfolio_state.open_positions)}")
                print(f"  - Daily PnL: ${portfolio_state.daily_pnl:,.2f}")
                print(f"  - Total equity: ${portfolio_state.equity:,.2f}")
                return portfolio_state
            
            else:
                print(f"No existing state found - creating fresh portfolio state")
                return PortfolioState(total_capital=default_capital)
        
        except Exception as e:
            print(f"ERROR loading portfolio state: {e}")
            print(f"Creating fresh portfolio state")
            return PortfolioState(total_capital=default_capital)
    
    def clear_state(self) -> bool:
        """
        Clear persisted state (useful for testing or reset).
        
        Returns:
            bool: True if successful
        """
        try:
            if self.storage_type == "redis":
                self.redis_client.delete(self.redis_key)
            
            elif self.storage_type == "file":
                if self.state_file_path.exists():
                    self.state_file_path.unlink()
            
            return True
        
        except Exception as e:
            print(f"ERROR clearing state: {e}")
            return False
    
    def create_backup(self, backup_suffix: Optional[str] = None) -> bool:
        """
        Create a timestamped backup of current state.
        
        Args:
            backup_suffix: Optional suffix for backup filename
        
        Returns:
            bool: True if successful
        """
        try:
            if backup_suffix is None:
                backup_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if self.storage_type == "redis":
                data = self.redis_client.get(self.redis_key)
                if data:
                    backup_path = Path(f"logs/rpm_state_backup_{backup_suffix}.json")
                    backup_path.parent.mkdir(parents=True, exist_ok=True)
                    with open(backup_path, 'w') as f:
                        f.write(data.decode('utf-8'))
                    return True
            
            elif self.storage_type == "file":
                if self.state_file_path.exists():
                    backup_path = self.state_file_path.with_name(
                        f"rpm_state_backup_{backup_suffix}.json"
                    )
                    import shutil
                    shutil.copy2(self.state_file_path, backup_path)
                    return True
            
            return False
        
        except Exception as e:
            print(f"ERROR creating backup: {e}")
            return False


class AutoSaveStateManager(StateManager):
    """
    State manager with automatic periodic saves.
    
    Wraps StateManager and saves state every N operations.
    """
    
    def __init__(
        self,
        storage_type: str = "file",
        redis_url: Optional[str] = None,
        state_file_path: Optional[str] = None,
        auto_save_frequency: int = 10,
    ):
        """
        Initialize auto-save state manager.
        
        Args:
            storage_type: "redis" or "file"
            redis_url: Redis connection URL
            state_file_path: Path to JSON state file
            auto_save_frequency: Save every N operations
        """
        super().__init__(storage_type, redis_url, state_file_path)
        self.auto_save_frequency = auto_save_frequency
        self.operation_counter = 0
    
    def maybe_save(self, portfolio_state: PortfolioState) -> bool:
        """
        Save if operation counter threshold reached.
        
        Args:
            portfolio_state: Current portfolio state
        
        Returns:
            bool: True if saved
        """
        self.operation_counter += 1
        
        if self.operation_counter >= self.auto_save_frequency:
            result = self.save_state(portfolio_state)
            self.operation_counter = 0
            return result
        
        return False
