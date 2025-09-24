# strategies/three_amigos.py
from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import pandas as pd

from .base import BaseStrategy


# ---------- Wilder helpers (kept consistent with other strategies) ----------
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


def _rsi_wilder(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = _wilder_rma(pd.Series(gain, index=close.index), period)
    avg_loss = _wilder_rma(pd.Series(loss, index=close.index), period)
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi


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


# ---------- Strategy ----------
class ThreeAmigos(BaseStrategy):
    """
    ENTRY #27 – “Three Amigos” (Kevin J. Davey) — Long side

    Idea (evaluated on the last COMPLETED bar):
      Eligible if:
        - ADX(adx_len) > adx_thresh
        - RSI(rsi_len) < rsi_mid
        - close < close.shift(lookback_big)
        - close > close.shift(lookback_short)

    Entry model:
      - Next-bar "market-like" entry. For engine compatibility we expose:
          HH := last close  (used as a stop-at-close proxy for fills)
          LL := LowestLow(lookback_short)  (simple protective anchor)
      - Position sizing uses trade_pct of equity.
      - Caps: max_positions, max_exposure_pct.

    Notes:
      • We only implement the long side to match the rest of the framework.
      • The paper-trader path expects HH/LL; we provide both accordingly.
    """

    def __init__(
        self,
        *,
        adx_len: int = 14,
        rsi_len: int = 14,
        lookback_big: int = 20,
        lookback_short: int = 10,
        adx_thresh: float = 25.0,
        rsi_mid: float = 50.0,
        trade_pct: float = 15.0,
        max_positions: int = 10,
        max_exposure_pct: float = 100.0,
        warmup_bars: int | None = None,
    ):
        # Params
        self.ADXLEN = int(adx_len)
        self.RSILEN = int(rsi_len)
        self.LB_BIG = int(lookback_big)
        self.LB_SHORT = int(lookback_short)
        self.ADXTHRESH = float(adx_thresh)
        self.RSIMID = float(rsi_mid)

        # Sizing & caps
        self.TRADE_PCT = float(trade_pct)
        self._max_positions = int(max_positions)
        self._max_exposure_pct = float(max_exposure_pct)

        # Warmup buffer for stable indicators & lookbacks
        default_warmup = max(
            self.ADXLEN * 5,
            self.RSILEN * 5,
            self.LB_BIG + 5,
            self.LB_SHORT + 5,
        )
        self._warmup_bars = default_warmup if warmup_bars is None else int(warmup_bars)

    @property
    def name(self) -> str:
        return "Three Amigos (Long)"

    # -------- indicators / dataframe prep --------
    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.dropna(subset=["open", "high", "low", "close"]).copy()

        # Core indicators
        d["RSI"] = _rsi_wilder(d["close"], self.RSILEN)
        d["ADX"] = _adx_wilder(d[["high", "low", "close"]], self.ADXLEN)

        # Momentum comparisons
        d["C_BIG"] = d["close"].shift(self.LB_BIG)
        d["C_SHORT"] = d["close"].shift(self.LB_SHORT)

        # Eligibility mask (long)
        d["ELIG"] = (
            (d["ADX"] > self.ADXTHRESH)
            & (d["RSI"] < self.RSIMID)
            & (d["close"] < d["C_BIG"])
            & (d["close"] > d["C_SHORT"])
        )

        # Compatibility with the engine/paper-trader:
        # - Treat next-bar market as a stop at the last close (HH).
        # - Use LL as a simple protective anchor from recent lows.
        d["HH"] = d["close"]
        d["LL"] = d["low"].rolling(self.LB_SHORT, min_periods=self.LB_SHORT).min()

        return d

    # -------- framework hooks --------
    def is_eligible(self, dfi: pd.Series) -> bool:
        elig = dfi.get("ELIG", False)
        return bool(elig) and pd.notna(dfi.get("HH")) and pd.notna(dfi.get("LL"))

    def next_entry_spec(self, symbol: str, df_i: pd.Series) -> Optional[Tuple[float, float]]:
        if not self.is_eligible(df_i):
            return None
        hh = float(df_i.get("HH", np.nan))
        ll = float(df_i.get("LL", np.nan))
        if not np.isfinite(hh) or hh <= 0:
            return None
        # If LL isn't ready yet, skip (protective anchor must be valid)
        if not np.isfinite(ll) or ll <= 0:
            return None
        # Entry @ HH (stop-at-close proxy), protective @ LL
        return hh, ll

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
