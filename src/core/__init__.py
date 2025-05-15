"""
Core package for the StratVector trading framework.
"""

from .backtester import Backtester
from .data_loader import DataLoader

__all__ = ['Backtester', 'DataLoader'] 