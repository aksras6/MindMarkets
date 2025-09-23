# strategies/rsi_trigger.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseStrategy


# ---- shared-style helpers (mirror rsi_breakout) ----
def _wilder_rma(values: pd.Series, n: int) -> pd.Series:
    """Wilder-style moving average (RMA)."""
    arr = values.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if n <= 0 or len(arr) < n:
        return pd.Series(out, index=values.index)
    # Seed with simple mean of first n
    init = np.nanmean(arr[:n])
    out[n - 1] = init
    for i in range(n, len(arr)):
        out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
    return pd.Series(out, index=values.index)


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI (Wilder), 0..100."""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder_rma(pd.Series(gain, index=close.index), period)
    avg_loss = _wilder_rma(pd.Series(loss, index=close.index), period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


class RsiTrigger(BaseStrategy):
    """
    RSI Trigger (Kevin J. Davey) — structured to MATCH the RSI Breakout class.

    Long-only momentum/trigger:
      Eligibility (on last completed bar):
        - RSI(rsi_len) < rsi_thresh
        - Close > SMA(xbars)
      Entry (next bar):
        - Market/stop-at-close style: use last close as entry reference
        - Protective stop: rolling LowestLow(xbars) as a simple stop anchor
      Sizing: trade_pct of equity
      Caps: max_positions, max_exposure_pct

    Defaults follow the book’s example idea (rsi_len=5, rsi_thresh=80, xbars=5).
    Raise rsi_thresh to be more selective (e.g., 70), or lower to be looser.
    """

    def __init__(
        self,
        *,
        rsi_len: int = 5,
        rsi_thresh: float = 80.0,
        xbars: int = 5,
        trade_pct: float = 15.0,
        max_positions: int = 10,
        max_exposure_pct: float = 100.0,
        warmup_bars: int | None = None,
    ):
        # Parameters (mirroring naming/casing style in RsiBreakout)
        self.RSILEN = int(rsi_len)
        self.RSITHRESH = float(rsi_thresh)
        self.XBARS = int(xbars)

        self.TRADE_PCT = float(trade_pct)
        self._max_positions = int(max_positions)
        self._max_exposure_pct = float(max_exposure_pct)

        # Warmup consistent with Breakout’s approach (ample buffer for stability)
        self._warmup_bars = (
            max(self.RSILEN * 5, self.XBARS + 5)
            if warmup_bars is None
            else int(warmup_bars)
        )

    @property
    def name(self) -> str:
        return "RSI Trigger (Long)"

    # -------- indicators / prep (match method name & style) --------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.dropna(subset=["open", "high", "low", "close"]).copy()
        d["RSI"] = _rsi_wilder(d["close"], self.RSILEN)
        d["SMA_X"] = d["close"].rolling(self.XBARS, min_periods=self.XBARS).mean()
        d["LLX"] = d["low"].rolling(self.XBARS, min_periods=self.XBARS).min()  # stop anchor
        return d

    # -------- eligibility (same signature as breakout) --------
    def is_eligible(self, dfi: pd.Series) -> bool:
        rsi = dfi.get("RSI", np.nan)
        sma = dfi.get("SMA_X", np.nan)
        close = dfi.get("close", np.nan)
        if not (pd.notna(rsi) and pd.notna(sma) and pd.notna(close)):
            return False
        # Davey trigger: RSI below threshold (i.e., not overbought) AND price above short SMA
        return (rsi < self.RSITHRESH) and (close > sma)

    # -------- entry spec (same return contract as breakout) --------
    def next_entry_spec(
        self, symbol: str, df_i: pd.Series
    ) -> Optional[Tuple[float, float]]:
        """
        Return (entry_price, protective_stop).
        We use last close as the reference entry (engine can treat as market/stop-at-close).
        Stop uses LLX over xbars as a basic protective level.
        """
        close = df_i.get("close", np.nan)
        llx = df_i.get("LLX", np.nan)
        if pd.isna(close) or close <= 0:
            return None
        # If LLX isn't ready yet, defer
        if pd.isna(llx) or llx <= 0:
            return None
        return float(close), float(llx)

    # -------- sizing (mirrors breakout) --------
    def dollars_per_trade(self, equity: float) -> float:
        return equity * (self.TRADE_PCT / 100.0)

    def shares_for_entry(self, entry_price: float, equity: float) -> int:
        if entry_price <= 0:
            return 0
        return int(self.dollars_per_trade(equity) // entry_price)

    # -------- caps & warmup (mirrors breakout) --------
    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def max_exposure_pct(self) -> float:
        return self._max_exposure_pct

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars
