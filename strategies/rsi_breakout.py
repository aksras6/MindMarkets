from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseStrategy

def _wilder_rma(values: pd.Series, n: int) -> pd.Series:
    """Wilder-style moving average (RMA)."""
    arr = values.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if n <= 0 or len(arr) < n:
        return pd.Series(out, index=values.index)
    # seed with simple mean of first n
    init = np.nanmean(arr[:n])
    out[n-1] = init
    for i in range(n, len(arr)):
        out[i] = (out[i-1]*(n-1) + arr[i]) / n
    return pd.Series(out, index=values.index)

def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    """Classic RSI (Wilder)."""
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder_rma(pd.Series(gain, index=close.index), period)
    avg_loss = _wilder_rma(pd.Series(loss, index=close.index), period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

class RsiBreakout(BaseStrategy):
    """
    Long-only breakout when momentum is healthy:
      - Eligibility: RSI(rsi_len) of the last *completed* bar >= rsi_thresh
      - Entry (next bar): Buy stop at HighestHigh(len_channel)
      - Protective stop: LowestLow(len_channel)
      - Sizing: trade_pct of equity
      - Caps: max_positions, max_exposure_pct

    Defaults favor a mild trend filter (RSI >= 55). Raise to 60-65 to be stricter.
    """

    def __init__(
        self,
        *,
        len_channel: int = 20,
        rsi_len: int = 14,
        rsi_thresh: float = 55.0,
        trade_pct: float = 15.0,
        max_positions: int = 10,
        max_exposure_pct: float = 100.0,
        warmup_bars: int | None = None
    ):
        self.LEN = int(len_channel)
        self.RSILEN = int(rsi_len)
        self.RSITHRESH = float(rsi_thresh)
        self.TRADE_PCT = float(trade_pct)
        self._max_positions = int(max_positions)
        self._max_exposure_pct = float(max_exposure_pct)
        # generous warmup for stable RSI & channels
        self._warmup_bars = (max(self.RSILEN * 5, self.LEN + 5)
                             if warmup_bars is None else int(warmup_bars))

    @property
    def name(self) -> str:
        return "RSI Breakout (Long)"

    # -------- indicators --------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.dropna(subset=["open","high","low","close"]).copy()
        d["RSI"] = _rsi_wilder(d["close"], self.RSILEN)
        d["HH"] = d["high"].rolling(self.LEN, min_periods=self.LEN).max()
        d["LL"] = d["low"].rolling(self.LEN, min_periods=self.LEN).min()
        return d

    # -------- logic --------
    def is_eligible(self, dfi: pd.Series) -> bool:
        rsi = dfi.get("RSI", np.nan)
        return (pd.notna(rsi) and rsi >= self.RSITHRESH)

    def next_entry_spec(self, symbol: str, df_i: pd.Series) -> Optional[Tuple[float, float]]:
        hh = df_i.get("HH", np.nan)
        ll = df_i.get("LL", np.nan)
        if pd.isna(hh) or pd.isna(ll) or hh <= 0:
            return None
        # buy stop at HH, protective stop at LL
        return float(hh), float(ll)

    # -------- sizing --------
    def dollars_per_trade(self, equity: float) -> float:
        return equity * (self.TRADE_PCT / 100.0)

    def shares_for_entry(self, entry_price: float, equity: float) -> int:
        if entry_price <= 0:
            return 0
        return int(self.dollars_per_trade(equity) // entry_price)

    # -------- caps & warmup --------
    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def max_exposure_pct(self) -> float:
        return self._max_exposure_pct

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars
