# strategies/three_amigos.py
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass
from .base import BaseStrategy

def _wilder_rma(x: pd.Series, n: int) -> pd.Series:
    x = x.astype(float)
    rma = x.ewm(alpha=1.0/n, adjust=False).mean()
    return rma

def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    up = np.where(delta > 0, delta, 0.0)
    dn = np.where(delta < 0, -delta, 0.0)
    rs = _wilder_rma(pd.Series(up, index=close.index), length) / \
         _wilder_rma(pd.Series(dn, index=close.index), length)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return pd.Series(rsi, index=close.index)

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    # Wilder’s DMI/ADX
    up_move   = high.diff()
    down_move = -low.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low  - close.shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    atr = _wilder_rma(tr, length)
    plus_di  = 100.0 * _wilder_rma(pd.Series(plus_dm,  index=close.index), length) / atr
    minus_di = 100.0 * _wilder_rma(pd.Series(minus_dm, index=close.index), length) / atr
    dx = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
    return _wilder_rma(dx, length)

@dataclass
class ThreeAmigos(BaseStrategy):
    """
    ENTRY #27 – “Three Amigos” (Kevin J. Davey)
      Long:  ADX>adx_thresh AND RSI<rsi_mid AND Close<Close[lookback_big] AND Close>Close[lookback_short]
      Short: ADX>adx_thresh AND RSI>rsi_mid AND Close>Close[lookback_big] AND Close<Close[lookback_short]
      Entries are next-bar market.
    """
    adx_len: int = 14
    rsi_len: int = 14
    lookback_big: int = 20
    lookback_short: int = 10
    adx_thresh: float = 25.0
    rsi_mid: float = 50.0
    warmup_bars: int = 80  # for GUI/backtester compatibility

    name: str = "three_amigos"

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Expected df columns: ['open','high','low','close', ...], indexed by datetime.
        Returns a DataFrame with boolean masks and next-bar market entry hints.
        """
        out = df.copy()
        c, h, l = out['close'], out['high'], out['low']

        out['_RSI'] = _rsi(c, self.rsi_len)
        out['_ADX'] = _adx(h, l, c, self.adx_len)

        # momentum comparisons
        big  = c.shift(self.lookback_big)
        shrt = c.shift(self.lookback_short)

        long_mask  = (out['_ADX'] > self.adx_thresh) & (out['_RSI'] < self.rsi_mid) & (c < big) & (c > shrt)
        short_mask = (out['_ADX'] > self.adx_thresh) & (out['_RSI'] > self.rsi_mid) & (c > big) & (c < shrt)

        # Your engine reads *_mkt=True to enter next bar at market (same as RSI Trigger we added)
        out['long_mkt']  = long_mask
        out['short_mkt'] = short_mask

        # (Optional) expose indicators for reporting/plotting
        out['indicator_adx'] = out['_ADX']
        out['indicator_rsi'] = out['_RSI']

        return out
