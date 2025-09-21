# Pluggable Strategy Module (Drop‑in)

This package gives you a clean `BaseStrategy` interface and one concrete strategy
(`AdxSqueezeBreakout`) that mirrors your current rules. You can import these
into your existing IB script and keep all IB/TWS plumbing unchanged.

## Folder layout

```
strategies/
  __init__.py
  base.py
  adx_squeeze.py
example_strategy_main.py  # small demo without IB
```

## How to integrate with your `ibapi_appv1.py`

1) Add imports near the top:
```python
from strategies.base import BaseStrategy
from strategies.adx_squeeze import AdxSqueezeBreakout
```

2) Instantiate the strategy once (e.g. in `__main__` or wherever you configure params):
```python
strat = AdxSqueezeBreakout(
    len_channel=20, adx_len=15, adx_thresh=20.0,
    trade_pct=15.0, max_positions=10, max_exposure_pct=100.0
)
```

3) When you fetch your latest completed daily dataframe for a symbol,
   pass it through `strategy.prepare(df)` **once** before using indicator columns.

4) Replace any hard-coded HH/LL/ADX checks with the strategy calls:
```python
last = prepared_df.iloc[-1]         # last completed bar
if strat.is_eligible(last):
    spec = strat.next_entry_spec(symbol, last)
    if spec:
        entry_stop, protective_stop = spec
        qty = strat.shares_for_entry(entry_stop, equity=ACCOUNT_SIZE)
        # Place your parent BUY stop @ entry_stop and child protective SELL stop @ protective_stop
```

5) For backtests, run `prepare` once per symbol and then loop over bars.
   Respect `strat.warmup_bars` before attempting entries.

## Try the demo (no IB required)

```bash
python example_strategy_main.py
```

## Adding a new strategy later

Create `strategies/my_new_strategy.py` with:

```python
from strategies.base import BaseStrategy
import pandas as pd

class MyNewStrategy(BaseStrategy):
    @property
    def name(self): return "My New Strategy"

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        # add indicators here
        return d

    def is_eligible(self, dfi: pd.Series) -> bool:
        return True  # your condition

    def next_entry_spec(self, symbol, df_i):
        return 123.45, 120.00  # (entry_stop, protective_stop)

    def dollars_per_trade(self, equity: float) -> float:
        return equity * 0.1

    def shares_for_entry(self, entry_price: float, equity: float) -> int:
        return int(self.dollars_per_trade(equity) // max(entry_price, 1e-9))

    @property
    def max_positions(self): return 10
    @property
    def max_exposure_pct(self): return 100.0
    @property
    def warmup_bars(self): return 50
```
