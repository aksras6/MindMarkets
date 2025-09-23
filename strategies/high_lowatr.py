from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseStrategy


def _wilder_rma(values: pd.Series, n: int) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) == 0 or n <= 0 or len(arr) < n:
        return pd.Series(out, index=values.index)
    init = np.nanmean(arr[:n])
    out[n - 1] = init
    for i in range(n, len(arr)):
        out[i] = (out[i - 1] * (n - 1) + arr[i]) / n
    return pd.Series(out, index=values.index)


def _atr_wilder(df_hlc: pd.DataFrame, period: int = 14) -> pd.Series:
    """Wilder ATR using the provided _wilder_rma helper."""
    h = df_hlc["high"].astype(float)
    l = df_hlc["low"].astype(float)
    c = df_hlc["close"].astype(float)
    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return _wilder_rma(tr, period)


class HighLowAtrBreakout(BaseStrategy):
    """
    Long signals on Tue/Wed/Thu when:
      - high is the highest high of last bbars, AND
      - close is the highest close of last bbars, AND
      - ATR(period) * big_point_value < maxl
    Entry is assumed next-bar @ market; we return (close, protective_stop).
    Protective stop is not defined by the original TS rule, so we return np.nan.
    """

    def __init__(
        self,
        *,
        bbars: int = 15,
        atr_len: int = 14,
        maxl: float = 2500.0,
        big_point_value: float = 1.0,
        trade_pct: float = 1.0,
        max_positions: int = 1,
        max_exposure_pct: float = 10.0,
        allowed_weekdays: tuple[int, ...] = (1, 2, 3),  # Tue/Wed/Thu (Mon=0)
        warmup_bars: int | None = None,
    ):
        self.BBARS = int(bbars)
        self.ATRLEN = int(atr_len)
        self.MAXL = float(maxl)
        self.BPV = float(big_point_value)
        self.TRADE_PCT = float(trade_pct)
        self._max_positions = int(max_positions)
        self._max_exposure_pct = float(max_exposure_pct)
        self.ALLOWED_WD = tuple(int(x) for x in allowed_weekdays)
        self._warmup_bars = (
            max(self.BBARS, self.ATRLEN) if warmup_bars is None else int(warmup_bars)
        )

    @property
    def name(self) -> str:
        return "High/Low ATR Breakout (Long)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.dropna(subset=["open", "high", "low", "close"])
        # Indicators
        d["ATR"] = _atr_wilder(d[["high", "low", "close"]], self.ATRLEN)
        d["HH_b"] = d["high"].rolling(self.BBARS, min_periods=self.BBARS).max()
        d["HC_b"] = d["close"].rolling(self.BBARS, min_periods=self.BBARS).max()
        d["weekday"] = d.index.dayofweek  # Monday=0
        # ATR in dollars per contract
        d["ATR_dollars"] = d["ATR"] * self.BPV
        return d

    def is_eligible(self, dfi: pd.Series) -> bool:
        # Weekday filter + ATR dollars threshold
        wd_ok = int(dfi.get("weekday", -1)) in self.ALLOWED_WD
        atr_d = dfi.get("ATR_dollars", np.nan)
        atr_ok = pd.notna(atr_d) and atr_d < self.MAXL
        return wd_ok and atr_ok

    def next_entry_spec(
        self, symbol: str, df_i: pd.Series
    ) -> Optional[Tuple[float, float]]:
        # Need rolling refs
        hh = df_i.get("HH_b", np.nan)
        hc = df_i.get("HC_b", np.nan)
        h = df_i.get("high", np.nan)
        c = df_i.get("close", np.nan)

        if not self.is_eligible(df_i):
            return None
        if any(pd.isna(x) for x in (hh, hc, h, c)):
            return None

        # TradeStation used equality checks; allow >= to be robust against float rounding.
        if (h >= hh) and (c >= hc):
            # Next bar at market -> use close as the "entry_stop" placeholder
            return float(c), float("nan")

        return None

    def dollars_per_trade(self, equity: float) -> float:
        return equity * (self.TRADE_PCT / 100.0)

    def shares_for_entry(self, entry_price: float, equity: float) -> int:
        if entry_price <= 0:
            return 0
        return int(self.dollars_per_trade(equity) // entry_price)

    @property
    def max_positions(self) -> int:
        return self._max_positions

    @property
    def max_exposure_pct(self) -> float:
        return self._max_exposure_pct

    @property
    def warmup_bars(self) -> int:
        return self._warmup_bars