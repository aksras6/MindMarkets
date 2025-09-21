from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple
import pandas as pd

@dataclass
class PositionSpec:
    """What to place for a new position on the next bar."""
    symbol: str
    qty: int
    entry_stop: float
    protective_stop: float

class BaseStrategy(ABC):
    """
    Contract a strategy must implement so main/backtester/paper-trader can plug-and-play.
    All prices are NEXT-BAR execution/placement decisions based on *completed* bar i.
    """

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of df with indicator columns added."""
        ...

    @abstractmethod
    def is_eligible(self, dfi: pd.Series) -> bool:
        """True if symbol is eligible to attempt entry after bar i closes."""
        ...

    @abstractmethod
    def next_entry_spec(self, symbol: str, df_i: pd.Series) -> Optional[Tuple[float, float]]:
        """Return (entry_stop, protective_stop) for next bar, or None."""
        ...

    @abstractmethod
    def dollars_per_trade(self, equity: float) -> float: ...

    @abstractmethod
    def shares_for_entry(self, entry_price: float, equity: float) -> int: ...

    @property
    @abstractmethod
    def max_positions(self) -> int: ...

    @property
    @abstractmethod
    def max_exposure_pct(self) -> float: ...

    @property
    @abstractmethod
    def warmup_bars(self) -> int: ...
