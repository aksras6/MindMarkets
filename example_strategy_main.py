"""
Demo runner to show the strategy interface in action.
This does NOT connect to IB. It synthesizes some data and shows how to call the strategy.
Integrate the same calls (`prepare`, `is_eligible`, `next_entry_spec`, `shares_for_entry`)
inside your ibapi script where you place/cancel orders.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from strategies.adx_squeeze import AdxSqueezeBreakout

def _fake_ohlc(n=200, start=100.0, seed=7):
    rng = np.random.default_rng(seed)
    rets = rng.normal(0, 0.01, n)
    close = start * np.exp(np.cumsum(rets))
    high = close * (1 + rng.uniform(0.0, 0.01, n))
    low = close * (1 - rng.uniform(0.0, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.002, n))
    idx = pd.date_range(datetime.now() - timedelta(days=n), periods=n, freq="D")
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)

def main():
    df = _fake_ohlc()
    strat = AdxSqueezeBreakout()
    d = strat.prepare(df)

    last = d.iloc[-1]
    eligible = strat.is_eligible(last)
    spec = strat.next_entry_spec("DEMO", last) if eligible else None

    print(f"Strategy: {strat.name}")
    print(f"Eligible now? {eligible}")
    if spec:
        entry, stop = spec
        qty = strat.shares_for_entry(entry, equity=10_000)
        print(f"Next BUY STOP {qty}@{entry:.2f} with protective STOP @{stop:.2f}")
    else:
        print("No entry spec for next bar.")

if __name__ == "__main__":
    main()
