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


def _adx_wilder(df_hlc: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df_hlc["high"]
    l = df_hlc["low"]
    c = df_hlc["close"]
    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = c.shift(1)
    tr = pd.concat(
        [(h - l).abs(), (h - prev_close).abs(), (l - prev_close).abs()], axis=1
    ).max(axis=1)

    atr = _wilder_rma(tr, period)
    plus_di = 100.0 * (_wilder_rma(pd.Series(plus_dm, index=c.index), period) / atr)
    minus_di = 100.0 * (_wilder_rma(pd.Series(minus_dm, index=c.index), period) / atr)

    dx = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
    adx = _wilder_rma(dx, period)
    return adx


class AdxSqueezeBreakout(BaseStrategy):
    def __init__(
        self,
        *,
        len_channel: int = 20,
        adx_len: int = 15,
        adx_thresh: float = 20.0,
        trade_pct: float = 15.0,
        max_positions: int = 10,
        max_exposure_pct: float = 100.0,
        warmup_bars: int = None,
    ):
        self.LEN = int(len_channel)
        self.ADXLEN = int(adx_len)
        self.ADXTHRESH = float(adx_thresh)
        self.TRADE_PCT = float(trade_pct)
        self._max_positions = int(max_positions)
        self._max_exposure_pct = float(max_exposure_pct)
        self._warmup_bars = (
            max(self.ADXLEN * 5, self.LEN + 5)
            if warmup_bars is None
            else int(warmup_bars)
        )

    @property
    def name(self) -> str:
        return "ADX Squeeze Breakout (Long)"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d = d.dropna(subset=["open", "high", "low", "close"])
        d["ADX"] = _adx_wilder(d[["high", "low", "close"]], self.ADXLEN)
        d["HH"] = d["high"].rolling(self.LEN, min_periods=self.LEN).max()
        d["LL"] = d["low"].rolling(self.LEN, min_periods=self.LEN).min()
        return d

    def is_eligible(self, dfi: pd.Series) -> bool:
        adx = dfi.get("ADX", np.nan)
        return pd.notna(adx) and adx < self.ADXTHRESH

    def next_entry_spec(
        self, symbol: str, df_i: pd.Series
    ) -> Optional[Tuple[float, float]]:
        hh = df_i.get("HH", np.nan)
        ll = df_i.get("LL", np.nan)
        if pd.isna(hh) or pd.isna(ll) or hh <= 0:
            return None
        return float(hh), float(ll)

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
