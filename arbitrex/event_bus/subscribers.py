"""
Event subscribers
"""

from typing import Callable, Optional, List
from .core import Event

class EventSubscriber:
    """Event subscriber with filtering"""
    
    def __init__(self, callback: Callable[[Event], None], symbols: Optional[List[str]] = None):
        self.callback = callback
        self.symbols = set(symbols) if symbols else None
        self.events_received = 0
    
    def matches(self, event: Event) -> bool:
        """Check if event matches filters"""
        if self.symbols and event.symbol and event.symbol not in self.symbols:
            return False
        return True
    
    def notify(self, event: Event):
        """Notify subscriber"""
        if self.matches(event):
            self.callback(event)
            self.events_received += 1
