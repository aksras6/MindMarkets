
"""
ADX Squeeze Breakout (Long) – IBKR data + portfolio backtest + CSV exports + Paper-Trading “dry run”

- Idempotent order placement (won't duplicate existing matching brackets)
- After-close daily roll can skip or cancel/replace intelligently

Mirrors the provided RealTest model (entry/exit/position sizing/caps/fees).

Rules (Daily):
- Compute ADX(15). If latest COMPLETED bar's ADX < 20, trading is allowed.
- Entry (next bar): Buy stop at HighestHigh(20) of completed bars.
- Exit (next bar): Sell stop at LowestLow(20) of completed bars.
- Position sizing: Percent-of-equity (TradePct = 15%).
- Portfolio caps: MaxPositions = 10, MaxExposure = 100%.
- Costs: Commission $0.01/share, Slippage $0.02/share.

CSV exports (to ./output):
- adx_trades.csv
- adx_equity.csv
- adx_kpis_by_symbol.csv
- adx_kpis_portfolio.csv
"""

from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract, ContractDetails
from ibapi.common import BarData
from ibapi.order import Order

import threading
import sys
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime, timezone
import os
from types import SimpleNamespace


# === Strategy interfaces ===
from strategies.base import BaseStrategy
from strategies.adx_squeeze import AdxSqueezeBreakout
from strategies.rsi_breakout import RsiBreakout


# --- Strategy registry for easy selection/extension ---
STRATEGY_REGISTRY = {
    "adx_squeeze": lambda **kw: AdxSqueezeBreakout(**kw),
    "rsi_breakout": lambda **kw: RsiBreakout(**kw),   # <-- add this line
    # "my_new": lambda **kw: MyNewStrategy(**kw),
}

# Which kwargs each strategy accepts (allow-list)
STRATEGY_PARAM_KEYS = {
    "adx_squeeze": {
        "len_channel", "adx_len", "adx_thresh",
        "trade_pct", "max_positions", "max_exposure_pct", "warmup_bars",
    },
    "rsi_breakout": {
        "len_channel", "rsi_len", "rsi_thresh",
        "trade_pct", "max_positions", "max_exposure_pct", "warmup_bars",
    },
}


def build_strategy(name: str, **overrides) -> BaseStrategy:
    key = (name or "adx_squeeze").strip().lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Known: {', '.join(STRATEGY_REGISTRY)}")

    allowed = STRATEGY_PARAM_KEYS.get(key, set())
    clean = {k: v for k, v in overrides.items() if k in allowed and v is not None}
    return STRATEGY_REGISTRY[key](**clean)


# =========================== Parameters (RealTest parity) ===========================
LEN = 20                 # channel lookback
ADXLEN = 15
ADXTHRESH = 20.0
TRADE_PCT = 15.0         # % of equity per position
ACCOUNT_SIZE = 10_000.0  # starting equity
MAX_POSITIONS = 10
MAX_EXPOSURE_PCT = 100.0 # total gross exposure limit

COMMISSION_PER_SH = 0.01
SLIPPAGE_PER_SH = 0.02

BAR_SIZE = "1 day"
USE_RTH = 1
WHAT_TO_SHOW = "TRADES"

# If you want to mirror StartDate: set START_DATE (YYYY-MM-DD).
# The downloader converts this into a year-based duration (IB requires years for >365d).
START_DATE = "2020-01-01"   # or None to use fixed duration below

# Fallback fixed duration in YEARS (string format like "8 Y") if START_DATE is None
DURATION_YEARS = "8 Y"

# Warmup to stabilize ADX & channels
WARMUP_BARS = max(ADXLEN * 5, LEN + 5)

# Optional normalization for tricky US tickers (IB uses spaces for BRK classes)
_SYMBOL_NORMALIZE = {
    "BRK.A": "BRK A",
    "BRK.B": "BRK B",
}

# Idempotency tolerances
PRICE_TOL = 1e-4  # treat stops as equal if within this amount

# =========================== DF helpers ===========================
def _bar_to_dict(bar: BarData) -> dict:
    return {
        "date": getattr(bar, "date", None),
        "open": getattr(bar, "open", None),
        "high": getattr(bar, "high", None),
        "low": getattr(bar, "low", None),
        "close": getattr(bar, "close", None),
        "volume": getattr(bar, "volume", None),
        "barCount": getattr(bar, "barCount", None),
        "wap": getattr(bar, "average", getattr(bar, "wap", None)),
    }

def _parse_ib_date(date_str: str) -> pd.Timestamp:
    if not isinstance(date_str, str):
        return pd.NaT
    if len(date_str) == 8:
        return pd.to_datetime(date_str, format="%Y%m%d", errors="coerce")
    if len(date_str) == 17:
        return pd.to_datetime(date_str, format="%Y%m%d  %H:%M:%S", errors="coerce")
    return pd.to_datetime(date_str, errors="coerce")

def bars_to_dataframe(symbol: str, bars: List[BarData]) -> pd.DataFrame:
    rows = [_bar_to_dict(b) for b in bars]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["datetime"] = df["date"].map(_parse_ib_date)
        df = df.drop(columns=["date"])
        df = df.sort_values("datetime").set_index("datetime")
        # Fixed minor typo: "barCount," -> "barCount"
        for col in ["open", "high", "low", "close", "volume", "barCount"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        if "wap" in df.columns:
            df["wap"] = pd.to_numeric(df["wap"], errors="coerce")
        df.insert(0, "symbol", symbol)
    return df

def upsert_bars_into_df(df: pd.DataFrame, symbol: str, new_bars: List[BarData]) -> pd.DataFrame:
    add_df = bars_to_dataframe(symbol, new_bars)
    if df is None or df.empty:
        return add_df
    if add_df.empty:
        return df
    combined = pd.concat([df, add_df])
    combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    return combined

# --- Gateway wrapper for consistent start/stop ---
class IBGateway:
    def __init__(self, app: "IbApp", host="127.0.0.1", port=7497, client_id=17):
        self.app = app
        self.host = host
        self.port = port
        self.client_id = client_id
        self._reader = None

    def start(self, wait_sec: int = 10):
        print("[GW] Connecting…")
        self.app.connect(self.host, self.port, clientId=self.client_id)
        self._reader = threading.Thread(target=self.app.run, name="ibapi-reader", daemon=True)
        self._reader.start()
        if not self.app.connected_evt.wait(timeout=wait_sec):
            raise TimeoutError("Timed out waiting for IB connection (nextValidId).")
        return self

    def stop(self):
        try:
            if self.app.isConnected():
                print("[GW] Disconnecting…")
                self.app.disconnect()
        finally:
            if self._reader:
                self._reader.join(timeout=5)
                if self._reader.is_alive():
                    print("[GW][WARN] Reader thread did not exit cleanly.", file=sys.stderr)


# =========================== IB App ===========================
class IbApp(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)
        self.connected_evt = threading.Event()

        # IDs
        self._req_id_counter: Optional[int] = None

        # Historical
        self._hist_done_evt = threading.Event()
        self._current_hist_req_id: Optional[int] = None
        the_symbol = None
        self._current_symbol: Optional[str] = the_symbol
        self.hist_bars_by_symbol: Dict[str, List[BarData]] = {}

        # Contract details
        self._cd_lock = threading.Lock()
        self._cd_done_evt = threading.Event()
        self._cd_by_reqid: Dict[int, List[ContractDetails]] = {}

        # DataFrames
        self.dfs_by_symbol: Dict[str, pd.DataFrame] = {}

        # Open orders tracking (for cancel/replace flows)
        self._open_orders = []                    # type: List[tuple]
        self._open_orders_done_evt = threading.Event()

    # ---- Connection / IDs ----
    def nextValidId(self, orderId: int):
        self._req_id_counter = max(1000, orderId)
        self.connected_evt.set()

    def next_req_id(self) -> int:
        if self._req_id_counter is None:
            raise RuntimeError("nextValidId not received yet.")
        self._req_id_counter += 1
        return self._req_id_counter

    # ---- Errors ----
    def error(self, reqId=None, errorCode=None, errorString=None, *extra):
        print(f"[IB][err] reqId={reqId} code={errorCode}: {errorString}")
        if extra:
            print(f"[IB][err-extra] {extra}")

    # ---- ContractDetails ----
    def contractDetails(self, reqId: int, contractDetails: ContractDetails):
        with self._cd_lock:
            self._cd_by_reqid.setdefault(reqId, []).append(contractDetails)

    def contractDetailsEnd(self, reqId: int):
        self._cd_done_evt.set()

    # ---- Historical ----
    def historicalData(self, reqId: int, bar: BarData):
        if self._current_hist_req_id == reqId:
            symbol = self._current_symbol
            if symbol is None:
                return
            self.hist_bars_by_symbol.setdefault(symbol, []).append(bar)

    def historicalDataEnd(self, reqId: int, start: str, end: str):
        self._hist_done_evt.set()

    # ---- Open Orders (for cancel/replace) ----
    def openOrder(self, orderId, contract, order, orderState):
        self._open_orders.append((orderId, contract, order))

    def openOrderEnd(self):
        self._open_orders_done_evt.set()

# =========================== Contract Resolution ===========================
def normalize_symbol(sym: str) -> str:
    return _SYMBOL_NORMALIZE.get(sym, sym)

def build_base_stock(symbol: str) -> Contract:
    c = Contract()
    c.symbol = symbol
    c.secType = "STK"
    c.currency = "USD"
    c.exchange = "SMART"
    return c

def resolve_contract(app: IbApp, symbol: str, timeout_sec: int = 10) -> Optional[Contract]:
    """
    Resolve a stock contract but DO NOT assign primaryExchange/primaryExch.
    Some ibapi builds lack those slots; assigning will raise AttributeError.
    """
    sym = normalize_symbol(symbol)
    base = build_base_stock(sym)

    req_id = app.next_req_id()
    app._cd_done_evt.clear()
    with app._cd_lock:
        app._cd_by_reqid.pop(req_id, None)

    print(f"[APP] Resolving contract for {symbol} (normalized: {sym}) … reqId={req_id}")
    app.reqContractDetails(req_id, base)

    if not app._cd_done_evt.wait(timeout=timeout_sec):
        print(f"[APP][WARN] contractDetails timeout for {symbol}")
        return None

    with app._cd_lock:
        cds = app._cd_by_reqid.get(req_id, [])

    if not cds:
        print(f"[APP][WARN] No contract details for {symbol}")
        return None

    # Prefer a USD stock line
    chosen: Optional[ContractDetails] = None
    for cd in cds:
        c = cd.contract
        if c.secType == "STK" and c.currency == "USD":
            chosen = cd
            break
    if chosen is None:
        chosen = cds[0]

    src = chosen.contract

    # Build a lean Contract WITHOUT touching primaryExchange/primaryExch
    resolved = Contract()
    resolved.conId = src.conId
    resolved.secType = "STK"
    resolved.exchange = "SMART"
    resolved.currency = "USD"
    resolved.symbol = src.symbol
    # NOTE: do not set resolved.primaryExchange / resolved.primaryExch here

    print(f"[APP] Resolved {symbol} → conId={resolved.conId}, sym={resolved.symbol}")
    return resolved

# ---------- Order-time contract shim (fix for primaryExchange AttributeError) ----------
def order_contract(src: Contract) -> object:
    """
    Returns a duck-typed object with all attributes IB's protobuf path may read.
    Avoids AttributeError on builds where Contract lacks 'primaryExchange'.
    """
    # Try using the native object if it already has the attribute the client_utils expects.
    if hasattr(src, "primaryExchange"):
        return src

    # Otherwise, build a namespace with everything client_utils.createContractProto touches.
    return SimpleNamespace(
        conId=getattr(src, "conId", 0),
        symbol=getattr(src, "symbol", ""),
        secType=getattr(src, "secType", ""),
        lastTradeDateOrContractMonth=getattr(src, "lastTradeDateOrContractMonth", ""),
        strike=getattr(src, "strike", 0.0),
        right=getattr(src, "right", ""),
        multiplier=float(getattr(src, "multiplier", 0.0)) if getattr(src, "multiplier", None) else 0.0,
        exchange=getattr(src, "exchange", "SMART"),
        primaryExchange=getattr(src, "primaryExch", ""),  # map old -> new
        currency=getattr(src, "currency", "USD"),
        localSymbol=getattr(src, "localSymbol", ""),
        tradingClass=getattr(src, "tradingClass", ""),
        includeExpired=getattr(src, "includeExpired", False),
        secIdType=getattr(src, "secIdType", ""),
        secId=getattr(src, "secId", ""),
        comboLegs=None,
        deltaNeutralContract=None,
    )

# =========================== Duration helpers (years only for >365d) ===========================
def duration_from_startdate(start_date_str: Optional[str]) -> str:
    if not start_date_str:
        return DURATION_YEARS
    try:
        start = datetime.strptime(start_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        days = (now - start).days
        years = max(1, int(np.ceil(days / 365.0)))
        return f"{years} Y"
    except Exception:
        return DURATION_YEARS

# =========================== Historical Download (robust) ===========================
def fetch_hist_once(
    app: IbApp,
    symbol: str,
    contract: Contract,
    duration_str: str,
    bar_size: str,
    what_to_show: str,
    use_rth: int,
    end_datetime: str,
    timeout_sec: int = 90
) -> List[BarData]:
    app._current_symbol = symbol
    app._hist_done_evt.clear()
    app.hist_bars_by_symbol.setdefault(symbol, [])
    app.hist_bars_by_symbol[symbol].clear()

    req_id = app.next_req_id()
    app._current_hist_req_id = req_id

    print(f"[APP] [{symbol}] Historical: dur={duration_str}, bar={bar_size}, show={what_to_show}, RTH={use_rth} (req_id={req_id})")
    app.reqHistoricalData(
        reqId=req_id,
        contract=contract,
        endDateTime=end_datetime,
        durationStr=duration_str,
        barSizeSetting=bar_size,
        whatToShow=what_to_show,
        useRTH=use_rth,
        formatDate=1,
        keepUpToDate=False,
        chartOptions=[]
    )

    if not app._hist_done_evt.wait(timeout=timeout_sec):
        print(f"[APP][WARN] [{symbol}] historicalDataEnd timeout after {timeout_sec}s")

    return app.hist_bars_by_symbol.get(symbol, [])

def try_download_symbol(app: IbApp, symbol: str) -> pd.DataFrame:
    contract = resolve_contract(app, symbol)
    if contract is None:
        return pd.DataFrame()

    dur_primary = duration_from_startdate(START_DATE)  # e.g., "6 Y"
    ladders = [
        (dur_primary, "TRADES", USE_RTH),
        (dur_primary, "ADJUSTED_LAST", USE_RTH),
        ("5 Y", "TRADES", USE_RTH),
        ("5 Y", "ADJUSTED_LAST", USE_RTH),
        ("1 Y", "TRADES", USE_RTH),
        ("1 Y", "ADJUSTED_LAST", USE_RTH),
        ("1 Y", "TRADES", 0),
        ("1 Y", "ADJUSTED_LAST", 0),
    ]
    for dur, wts, rth in ladders:
        try:
            bars = fetch_hist_once(
                app,
                symbol=symbol,
                contract=contract,
                duration_str=dur,
                bar_size=BAR_SIZE,
                what_to_show=wts,
                use_rth=rth,
                end_datetime=""
            )
            df = upsert_bars_into_df(pd.DataFrame(), symbol, bars)
            if not df.empty and all(c in df.columns for c in ["open", "high", "low", "close"]):
                return df[["symbol","open","high","low","close","volume"]].dropna()
        except Exception as e:
            print(f"[APP][{symbol}][retry] {dur}/{wts}/RTH={rth} failed: {e}")
        time.sleep(0.5)  # gentle pacing
    print(f"[APP][{symbol}] No historical data after retries — skipping.")
    return pd.DataFrame()

# =========================== Indicators (Wilder) ===========================
def wilder_rma(values: pd.Series, n: int) -> pd.Series:
    arr = values.to_numpy(dtype=float)
    out = np.full_like(arr, np.nan, dtype=float)
    if len(arr) == 0 or n <= 0:
        return pd.Series(out, index=values.index)
    if len(arr) < n:
        return pd.Series(out, index=values.index)
    init = np.nanmean(arr[:n])
    out[n-1] = init
    for i in range(n, len(arr)):
        out[i] = (out[i-1]*(n-1) + arr[i]) / n
    return pd.Series(out, index=values.index)

def adx_wilder(df_hlc: pd.DataFrame, period: int = 14) -> pd.Series:
    h = df_hlc["high"]; l = df_hlc["low"]; c = df_hlc["close"]
    up_move = h.diff()
    down_move = -l.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    prev_close = c.shift(1)
    tr = pd.concat([
        (h - l).abs(),
        (h - prev_close).abs(),
        (l - prev_close).abs()
    ], axis=1).max(axis=1)

    atr = wilder_rma(tr, period)
    plus_di = 100.0 * (wilder_rma(pd.Series(plus_dm, index=c.index), period) / atr)
    minus_di = 100.0 * (wilder_rma(pd.Series(minus_dm, index=c.index), period) / atr)

    dx = 100.0 * (plus_di.subtract(minus_di).abs() / (plus_di + minus_di))
    adx = wilder_rma(dx, period)
    return adx

# =========================== Backtest structures ===========================
@dataclass
class Position:
    symbol: str
    shares: int
    entry_price: float
    entry_idx: int  # bar index in df
    equity_at_entry: float

@dataclass
class Trade:
    symbol: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    shares: int
    pnl: float
    ret_pct: float

# =========================== Execution price models ===========================
def stop_entry_fill_price(next_bar_open: float, next_bar_high: float, stop_price: float) -> Optional[float]:
    if np.isnan(next_bar_open) or np.isnan(next_bar_high) or np.isnan(stop_price):
        return None
    if next_bar_high >= stop_price:
        raw = max(stop_price, next_bar_open)
        return raw + SLIPPAGE_PER_SH
    return None

def stop_exit_fill_price(next_bar_open: float, next_bar_low: float, stop_price: float) -> Optional[float]:
    if np.isnan(next_bar_open) or np.isnan(next_bar_low) or np.isnan(stop_price):
        return None
    if next_bar_low <= stop_price:
        raw = min(stop_price, next_bar_open)
        return raw - SLIPPAGE_PER_SH
    return None

# =========================== Idempotent Order Utilities ===========================
def _fetch_open_orders(app: IbApp) -> List[tuple]:
    """Return list of (orderId, contract, order) that are currently open."""
    app._open_orders = []  # type: List[tuple]
    app._open_orders_done_evt.clear()
    app.reqOpenOrders()
    app._open_orders_done_evt.wait(timeout=10)
    return list(app._open_orders)

def _parents_by_symbol(open_orders: List[tuple]) -> Dict[str, List[Tuple[int, object, object]]]:
    """
    Index BUY stop parents (no parentId or parentId==0) by contract.symbol.
    Returns: { "AAPL": [(orderId, contract, order), ...], ... }
    """
    out: Dict[str, List[Tuple[int, object, object]]] = {}
    for oid, c, o in open_orders:
        sym = getattr(c, "symbol", "")
        if not sym:
            continue
        if getattr(o, "orderType", "") == "STP" and getattr(o, "action", "").upper() == "BUY":
            pid = getattr(o, "parentId", 0) or 0
            if pid == 0:
                out.setdefault(sym, []).append((oid, c, o))
    return out

def _cancel_bracket_by_parent(app: IbApp, parent_id: int, open_orders: List[tuple]) -> None:
    """Cancel parent and any children that reference it via parentId."""
    to_cancel = {oid for oid, _, _ in open_orders if oid == parent_id}
    for oid, _, o in open_orders:
        if getattr(o, "parentId", 0) == parent_id:
            to_cancel.add(oid)
    for oid in sorted(to_cancel):
        try:
            print(f"[ORD] Cancelling orderId={oid}")
            app.cancelOrder(oid)
        except Exception as e:
            print(f"[ORD] cancelOrder({oid}) error: {e}")

# =========================== Backtester ===========================
def backtest_portfolio(
    dfs_by_symbol: Dict[str, pd.DataFrame],
    strategy: BaseStrategy,
    ACCOUNT_SIZE: float = ACCOUNT_SIZE,
    COMMISSION_PER_SH: float = COMMISSION_PER_SH,
    SLIPPAGE_PER_SH: float = SLIPPAGE_PER_SH,
) -> Tuple[pd.DataFrame, Dict[str, any], pd.DataFrame]:

    dfs = {}
    for sym, df in dfs_by_symbol.items():
        d = strategy.prepare(df.copy())
        dfs[sym] = d

    if not dfs:
        return pd.DataFrame(), {}, pd.DataFrame()

    all_index = sorted(set().union(*[df.index for df in dfs.values()]))
    for sym in list(dfs.keys()):
        dfs[sym] = dfs[sym].reindex(all_index).ffill()

    syms = list(dfs.keys())
    n = len(all_index)

    equity = ACCOUNT_SIZE
    cash = ACCOUNT_SIZE
    positions: Dict[str, Position] = {}
    trades: List[Trade] = []
    equity_curve = np.zeros(n)

    start_idx = 0
    for i in range(n):
        if i >= strategy.warmup_bars:
            start_idx = i
            break

    equity_curve[:] = 0.0
    equity_curve[: start_idx + 1] = ACCOUNT_SIZE

    def stop_entry_fill_price(next_open, next_high, stop_price):
        if any(pd.isna(x) for x in (next_open, next_high, stop_price)):
            return None
        if next_high >= stop_price:
            raw = max(stop_price, next_open)
            return raw + SLIPPAGE_PER_SH
        return None

    def stop_exit_fill_price(next_open, next_low, stop_price):
        if any(pd.isna(x) for x in (next_open, next_low, stop_price)):
            return None
        if next_low <= stop_price:
            raw = min(stop_price, next_open)
            return raw - SLIPPAGE_PER_SH
        return None

    for i in range(start_idx, n - 1):
        next_i = i + 1

        to_close = []
        for sym, pos in positions.items():
            d = dfs[sym]
            stop_loss = d.get("LL").iloc[i] if "LL" in d.columns else None
            if stop_loss is None or pd.isna(stop_loss):
                continue
            nb_open = d["open"].iloc[next_i]
            nb_low  = d["low"].iloc[next_i]
            exit_fill = stop_exit_fill_price(nb_open, nb_low, stop_loss)
            if exit_fill is not None:
                comm = pos.shares * COMMISSION_PER_SH
                pnl = (exit_fill - pos.entry_price) * pos.shares - comm
                trades.append(Trade(
                    symbol=sym,
                    entry_date=all_index[pos.entry_idx],
                    exit_date=all_index[next_i],
                    entry_price=pos.entry_price,
                    exit_price=exit_fill,
                    shares=pos.shares,
                    pnl=pnl,
                    ret_pct=(exit_fill / pos.entry_price - 1.0) * 100.0
                ))
                cash += pos.shares * exit_fill - comm
                to_close.append(sym)
        for sym in to_close:
            positions.pop(sym, None)

        max_positions_left = strategy.max_positions - len(positions)
        gross = sum(dfs[s]["close"].iloc[i] * positions[s].shares for s in positions)
        exposure_pct = 100.0 * gross / max(equity, 1e-9)

        if max_positions_left > 0 and exposure_pct < strategy.max_exposure_pct - 1e-9:
            entry_candidates = []
            for sym in syms:
                if sym in positions:
                    continue
                d = dfs[sym]
                if not strategy.is_eligible(d.iloc[i]):
                    continue
                spec = strategy.next_entry_spec(sym, d.iloc[i])
                if spec is None:
                    continue
                entry_stop, protective_stop = spec
                nb_open = d["open"].iloc[next_i]
                nb_high = d["high"].iloc[next_i]
                entry_fill = stop_entry_fill_price(nb_open, nb_high, entry_stop)
                if entry_fill is None:
                    continue
                shares = strategy.shares_for_entry(entry_fill, equity)
                if shares <= 0:
                    continue
                comm = shares * COMMISSION_PER_SH
                cost = shares * entry_fill + comm
                if cost > cash + 1e-9:
                    continue
                entry_candidates.append((sym, shares, entry_fill, comm))

            for sym, shares, entry_fill, comm in entry_candidates:
                if len(positions) >= strategy.max_positions:
                    break
                gross_after = sum(dfs[s]["close"].iloc[i] * positions[s].shares for s in positions) + shares * entry_fill
                exposure_pct = 100.0 * gross_after / max(equity, 1e-9)
                if exposure_pct > strategy.max_exposure_pct + 1e-9:
                    continue
                cash -= shares * entry_fill + comm
                positions[sym] = Position(
                    symbol=sym,
                    shares=shares,
                    entry_price=entry_fill,
                    entry_idx=next_i,
                    equity_at_entry=equity
                )

        close_mv = sum(dfs[s]["close"].iloc[next_i] * positions[s].shares for s in positions)
        equity = cash + close_mv
        equity_curve[next_i] = equity

    final_i = n - 1
    for sym, pos in list(positions.items()):
        px = dfs[sym]["close"].iloc[final_i]
        comm = pos.shares * COMMISSION_PER_SH
        pnl = (px - pos.entry_price) * pos.shares - comm
        trades.append(Trade(
            symbol=sym,
            entry_date=all_index[pos.entry_idx],
            exit_date=all_index[final_i],
            entry_price=pos.entry_price,
            exit_price=px,
            shares=pos.shares,
            pnl=pnl,
            ret_pct=(px / pos.entry_price - 1.0) * 100.0
        ))
        cash += pos.shares * px - comm
        positions.pop(sym, None)
    equity = cash
    equity_curve[final_i] = equity

        # Equity & returns (use post-warmup segment to avoid flat early curve)
    eq = pd.Series(equity_curve, index=all_index).replace(0.0, np.nan).ffill().fillna(ACCOUNT_SIZE)
    eq_eff = eq.iloc[start_idx:]  # effective equity segment (after warmup)
    returns = (
        eq_eff.pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
    )

    periods = len(eq_eff)  # bars considered (post-warmup)
    start_equity_eff = float(eq_eff.iloc[0])
    final_equity = float(eq_eff.iloc[-1])

    net_profit = final_equity - start_equity_eff
    total_return_pct = (final_equity / start_equity_eff - 1.0) * 100.0

    # Annualised CAGR using trading-day convention (252)
    n_days = max(1, returns.shape[0])
    cagr_pct = ((final_equity / start_equity_eff) ** (252.0 / n_days) - 1.0) * 100.0

    # Max drawdown (positive percent)
    running_max = eq_eff.cummax()
    dd = (eq_eff - running_max) / running_max
    max_dd_pct = float(abs(dd.min()) * 100.0)


    trades_df = pd.DataFrame([t.__dict__ for t in trades]) if trades else pd.DataFrame(
        columns=["symbol","entry_date","exit_date","entry_price","exit_price","shares","pnl","ret_pct"]
    )

    n_trades = len(trades_df)
    wins = (trades_df["pnl"] > 0).sum() if n_trades else 0
    pct_wins = (wins / n_trades * 100.0) if n_trades else 0.0
    gross_profit = trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum() if n_trades else 0.0
    gross_loss = -trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum() if n_trades else 0.0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float("inf")

    ret_std = returns.std(ddof=1)
    ret_mean = returns.mean()
    sharpe = (np.sqrt(252.0) * ret_mean / ret_std) if (ret_std is not None and ret_std > 0) else np.nan

    # Annualised volatility (optional)
    ann_vol_pct = float(returns.std(ddof=1) * np.sqrt(252.0) * 100.0) if returns.shape[0] > 2 else np.nan

    summary = {
        "Strategy": strategy.name,
        "BarsUsed": periods,                      # post-warmup bars
        "StartEquity": round(float(start_equity_eff), 2),
        "FinalEquity": round(float(final_equity), 2),

        # Returns
        "TotalReturnPct": round(float(total_return_pct), 2),
        "CAGR": round(float(cagr_pct), 2),
        "ROR": round(float(total_return_pct), 2),  # backward-compat alias for TotalReturnPct

        # Risk
        "MaxDDPct": round(float(max_dd_pct), 2),
        "Calmar": (
            round(float((cagr_pct / max_dd_pct)) , 3)
            if (max_dd_pct > 0 and np.isfinite(cagr_pct)) else "NA"
        ),
        "Sharpe": round(float(sharpe), 3) if pd.notna(sharpe) else "NA",

        # Trading stats
        "Trades": int(n_trades),
        "PctWins": round(float(pct_wins), 2),
        "ProfitFactor": round(float(profit_factor), 2) if np.isfinite(profit_factor) else "Inf",

        # Original fields retained
        "NetProfit": round(float(net_profit), 2),

        "AnnVolPct": round(ann_vol_pct, 2) if pd.notna(ann_vol_pct) else "NA",

    }


    equity_df = pd.DataFrame({"date": all_index, "equity": eq.values})

    return trades_df, summary, equity_df

# =========================== Per-symbol KPI breakdown ===========================
# =========================== Per-symbol KPI breakdown ===========================
def kpis_by_symbol(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        cols = ["symbol","Trades","NetProfit","PctWins","ProfitFactor","AvgRetPct","MedianRetPct"]
        return pd.DataFrame(columns=cols)

    gp = trades_df.groupby("symbol", dropna=False)

    kpis = gp.agg(
        Trades=("pnl","count"),
        NetProfit=("pnl","sum"),
        Wins=("pnl", lambda s: (s > 0).sum() ),
        GrossProfit=("pnl", lambda s: s[s > 0].sum()),
        GrossLoss=("pnl", lambda s: -s[s < 0].sum()),
        AvgRetPct=("ret_pct","mean"),
        MedianRetPct=("ret_pct","median"),
    ).reset_index()

    kpis["PctWins"] = np.where(kpis["Trades"] > 0, 100.0 * kpis["Wins"] / kpis["Trades"], 0.0)
    kpis["ProfitFactor"] = np.where(kpis["GrossLoss"] > 0, kpis["GrossProfit"] / kpis["GrossLoss"], np.inf)

    kpis = kpis.drop(columns=["Wins","GrossProfit","GrossLoss"])
    kpis["NetProfit"] = kpis["NetProfit"].round(2)
    kpis["PctWins"] = kpis["PctWins"].round(2)
    kpis["ProfitFactor"] = kpis["ProfitFactor"].replace(np.inf, np.nan).round(2).fillna("Inf")
    kpis["AvgRetPct"] = kpis["AvgRetPct"].round(3)
    kpis["MedianRetPct"] = kpis["MedianRetPct"].round(3)

    return kpis[["symbol","Trades","NetProfit","PctWins","ProfitFactor","AvgRetPct","MedianRetPct"]]

# =========================== Save CSV helpers ===========================
def save_csvs(output_dir: str, trades_df: pd.DataFrame, summary: Dict[str, any], equity_df: pd.DataFrame):
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "adx_trades.csv")
    trades_df.to_csv(trades_path, index=False)

    equity_path = os.path.join(output_dir, "adx_equity.csv")
    equity_df.to_csv(equity_path, index=False)

    sym_kpis_df = kpis_by_symbol(trades_df)
    sym_kpis_path = os.path.join(output_dir, "adx_kpis_by_symbol.csv")
    sym_kpis_df.to_csv(sym_kpis_path, index=False)

    port_kpis_df = pd.DataFrame([summary])
    port_kpis_path = os.path.join(output_dir, "adx_kpis_portfolio.csv")
    port_kpis_df.to_csv(port_kpis_path, index=False)

    print(f"\\n[OUT] Saved:")
    print(f" - Trades:        {trades_path}")
    print(f" - Equity curve:  {equity_path}")
    print(f" - KPIs by sym:   {sym_kpis_path}")
    print(f" - KPIs portfolio:{port_kpis_path}")

# =========================== Orchestration (Historical Backtest) ===========================
def run_download_and_backtest(
    symbols: List[str],
    host: str = "127.0.0.1",
    port: int = 7497,
    client_id: int = 17,
    output_dir: str = "output"
):
    app = IbApp()
    reader = None
    try:
        print("[APP] Connecting to IB…")
        app.connect(host, port, clientId=client_id)
        reader = threading.Thread(target=app.run, name="ibapi-reader", daemon=False)
        reader.start()
        if not app.connected_evt.wait(timeout=10):
            raise TimeoutError("Timed out waiting for IB connection (nextValidId).")

        usable = {}
        for sym in symbols:
            print(f"[APP] Requesting daily history for {sym}…")
            df = try_download_symbol(app, sym)
            if df.empty:
                print(f"[APP] Skipping {sym} (no data).")
                continue
            usable[sym] = df

        if not usable:
            print("[APP] No symbols produced data. Check permissions/contracts and try again.")
            return pd.DataFrame(), {}

        print("[BT] Running backtest…")
        trades_df, summary, equity_df = backtest_portfolio(usable)

        print("\\n=== Backtest KPIs (Portfolio) ===")
        for k, v in summary.items():
            print(f"{k}: {v}")

        if not trades_df.empty:
            print("\\n=== First 10 trades ===")
            print(trades_df.head(10).to_string(index=False))
        else:
            print("\\n(No trades generated for the given range.)")

        save_csvs(output_dir, trades_df, summary, equity_df)
        return trades_df, summary

    finally:
        if app.isConnected():
            print("[APP] Disconnecting…")
            app.disconnect()
        if reader:
            reader.join(timeout=5)
            if reader.is_alive():
                print("[APP][WARN] Reader thread did not exit cleanly.", file=sys.stderr)

# =========================== Paper-Trading “Dry Run Now” (Idempotent) ===========================
def create_stop_order(action: str, total_qty: int, stop_price: float, tif: str="GTC") -> Order:
    o = Order()
    o.action = action.upper()
    o.orderType = "STP"
    o.totalQuantity = int(total_qty)
    o.auxPrice = float(stop_price)
    o.tif = tif
    o.transmit = True
    return o

def create_bracket_buy(parent_id: int, qty: int, entry_stop: float, stop_loss: float, tif: str="GTC"):
    parent = create_stop_order("BUY", qty, entry_stop, tif)
    parent.orderId = parent_id
    parent.transmit = False

    child = create_stop_order("SELL", qty, stop_loss, tif)
    child.parentId = parent_id
    child.orderId = parent_id + 1
    child.transmit = True
    return [parent, child]

def latest_completed_daily_df(app: IbApp, symbol: str, strategy: BaseStrategy) -> pd.DataFrame:
    """Fetch latest daily bars for `symbol` and return `strategy.prepare(df)` with indicators."""
    contract = resolve_contract(app, symbol)
    if contract is None:
        return pd.DataFrame()

    dur = duration_from_startdate(START_DATE)
    bars = fetch_hist_once(
        app, symbol, contract, dur, BAR_SIZE, WHAT_TO_SHOW, USE_RTH, end_datetime="", timeout_sec=90
    )
    df = upsert_bars_into_df(pd.DataFrame(), symbol, bars)
    if df.empty:
        return df

    d = df.dropna(subset=["open","high","low","close"]).copy()
    # Delegate indicator prep to the strategy
    d = strategy.prepare(d)
    return d


def place_paper_orders_now(
    symbols: List[str],
    strategy: BaseStrategy,
    host: str="127.0.0.1",
    port: int=7497,         # 7497 = PAPER (7496 = LIVE)
    client_id: int=44,
    equity: float=ACCOUNT_SIZE
):

    """
    Idempotent: checks existing open parent BUY STP brackets and skips if identical,
    otherwise cancels/replaces the whole bracket.
    """
    app = IbApp()
    reader = None
    try:
        print("[DRY] Connecting to IB Paper…")
        app.connect(host, port, clientId=client_id)
        reader = threading.Thread(target=app.run, name="ibapi-reader", daemon=True)
        reader.start()
        if not app.connected_evt.wait(10):
            raise TimeoutError("Timed out waiting for IB connection.")

        while app._req_id_counter is None:
            time.sleep(0.1)
        next_id = app._req_id_counter + 1

        # Snapshot currently open orders once
        open_orders = _fetch_open_orders(app)
        parents_map = _parents_by_symbol(open_orders)

        open_count = 0
        gross_target = 0.0

        for sym in symbols:
            if open_count >= strategy.max_positions:
                break

            d = latest_completed_daily_df(app, sym, strategy)
            if d.empty or len(d) < strategy.warmup_bars + 1:
                print(f"[DRY][{sym}] insufficient data; skipping")
                continue

            last = d.iloc[-1]
            hh = float(last.get("HH", float('nan'))); ll = float(last.get("LL", float('nan')))

            if any(pd.isna(x) for x in (hh, ll)):
                print(f"[DRY][{sym}] indicators NaN; skipping")
                continue
            if not strategy.is_eligible(last):
                #print(f"[DRY][{sym}] ADX={adx:.1f} ≥ {ADXTHRESH}; not eligible")
                continue

            qty = strategy.shares_for_entry(hh, equity)
            if hh <= 0:
                print(f"[DRY][{sym}] bad HH; skipping")
                continue
            if qty <= 0:
                print(f"[DRY][{sym}] qty=0 at stop {hh:.2f}; skipping")
                continue

            if (gross_target + qty * hh) / equity * 100.0 > strategy.max_exposure_pct + 1e-9:
                print(f"[DRY][{sym}] exposure cap would be exceeded; skipping")
                continue

            # Idempotency check
            existing_parents = parents_map.get(normalize_symbol(sym), []) or parents_map.get(sym, [])
            identical_found = False
            for ex_oid, _c, ex_o in existing_parents:
                ex_qty = int(getattr(ex_o, "totalQuantity", 0))
                ex_px  = float(getattr(ex_o, "auxPrice", 0.0))
                if ex_qty == qty and abs(ex_px - hh) <= PRICE_TOL:
                    print(f"[DRY][{sym}] matching bracket already exists (qty={qty}, stop={ex_px:.2f}); skipping")
                    identical_found = True
                    break
            if identical_found:
                continue

            # If something exists but differs, cancel full bracket(s) for this symbol
            for ex_oid, _c, _o in existing_parents:
                _cancel_bracket_by_parent(app, ex_oid, open_orders)

            base_contract = resolve_contract(app, sym)
            if base_contract is None:
                print(f"[DRY][{sym}] could not resolve contract; skipping")
                continue

            contract_for_order = order_contract(base_contract)

            oid = next_id
            bracket = create_bracket_buy(oid, qty, hh, ll, tif="GTC")
            next_id += len(bracket)

            print(f"[DRY][{sym}] placing BUY STOP {qty}@{hh:.2f} with protective SELL STOP @{ll:.2f}")
            for o in bracket:
                app.placeOrder(o.orderId, contract_for_order, o)

            open_count += 1
            gross_target += qty * hh

        print("[DRY] Done. Idempotent placement complete (see TWS Orders).")
    finally:
        if app.isConnected():
            print("[DRY] Disconnecting…")
            app.disconnect()
        if reader:
            reader.join(timeout=3)

# =========================== Daily "roll" after-close (cancel/replace but skip identical) ===========================
try:
    from zoneinfo import ZoneInfo
except Exception:
    ZoneInfo = None

def _after_us_close_now():
    if ZoneInfo is None:
        return True  # fall back to "always allow"
    now_et = datetime.now(ZoneInfo("America/New_York"))
    if now_et.weekday() >= 5:
        return False
    cutoff = now_et.replace(hour=16, minute=5, second=0, microsecond=0)
    return now_et >= cutoff


def roll_daily_brackets_after_close(
    symbols: List[str],
    strategy: BaseStrategy,
    host: str="127.0.0.1",
    port: int=7497,      # Paper
    client_id: int=55,
    equity: float=ACCOUNT_SIZE
):

    """
    After US close:
      - Fetch existing open orders once.
      - For each eligible symbol, if an identical BUY STP parent exists (qty & price ~equal), SKIP.
      - Otherwise cancel that symbol's existing parent bracket(s) and place a fresh bracket.
    """
    if not _after_us_close_now():
        print("[ROLL] Not after US close yet. Skipping.")
        return

    app = IbApp()
    reader = None
    try:
        print("[ROLL] Connecting to IB Paper…")
        app.connect(host, port, clientId=client_id)
        reader = threading.Thread(target=app.run, name="ibapi-reader", daemon=True)
        reader.start()
        if not app.connected_evt.wait(10):
            raise TimeoutError("Timed out waiting for IB connection.")

        while app._req_id_counter is None:
            time.sleep(0.1)
        next_id = app._req_id_counter + 1

        # Snapshot open orders once
        open_orders = _fetch_open_orders(app)
        parents_map = _parents_by_symbol(open_orders)

        open_count = 0
        gross_target = 0.0

        for sym in symbols:
            if open_count >= strategy.max_positions:
                break

            d = latest_completed_daily_df(app, sym, strategy)
            if d.empty or len(d) < strategy.warmup_bars + 1:
                print(f"[ROLL][{sym}] insufficient data; skipping")
                continue

            last = d.iloc[-1]
            hh = float(last.get("HH", float('nan'))); ll = float(last.get("LL", float('nan')))
            if any(pd.isna(x) for x in (hh, ll)):
                print(f"[ROLL][{sym}] NaN indicator(s); skipping")
                continue
            if not strategy.is_eligible(last):
                #print(f"[ROLL][{sym}] not eligible by {strategy.name}")
                continue

            qty = strategy.shares_for_entry(hh, equity)
            if hh <= 0:
                print(f"[ROLL][{sym}] bad HH; skipping")
                continue
            if qty <= 0:
                print(f"[ROLL][{sym}] qty=0 at stop {hh:.2f}; skipping")
                continue

            if (gross_target + qty * hh) / equity * 100.0 > strategy.max_exposure_pct + 1e-9:
                print(f"[ROLL][{sym}] exposure cap would be exceeded; skipping")
                continue

            # Idempotency: skip identical, else cancel and replace
            existing_parents = parents_map.get(normalize_symbol(sym), []) or parents_map.get(sym, [])
            identical_found = False
            for ex_oid, _c, ex_o in existing_parents:
                ex_qty = int(getattr(ex_o, "totalQuantity", 0))
                ex_px  = float(getattr(ex_o, "auxPrice", 0.0))
                if ex_qty == qty and abs(ex_px - hh) <= PRICE_TOL:
                    print(f"[ROLL][{sym}] matching bracket already exists (qty={qty}, stop={ex_px:.2f}); skipping")
                    identical_found = True
                    break
            if identical_found:
                continue

            # Cancel existing for this symbol (if any)
            for ex_oid, _c, _o in existing_parents:
                _cancel_bracket_by_parent(app, ex_oid, open_orders)

            base_contract = resolve_contract(app, sym)
            if base_contract is None:
                print(f"[ROLL][{sym}] contract resolution failed; skipping")
                continue

            contract_for_order = order_contract(base_contract)

            parent_id = next_id
            bracket = create_bracket_buy(parent_id, qty, hh, ll, tif="GTC")
            next_id += len(bracket)

            print(f"[ROLL][{sym}] placing BUY STOP {qty}@{hh:.2f} with protective SELL STOP @{ll:.2f}")
            for o in bracket:
                app.placeOrder(o.orderId, contract_for_order, o)

            open_count += 1
            gross_target += qty * hh

        print("[ROLL] Done. Check TWS → Orders.")
    finally:
        if app.isConnected():
            print("[ROLL] Disconnecting…")
            app.disconnect()
        if reader:
            reader.join(timeout=3)

# =========================== Main ===========================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="IBKR ADX Squeeze — backtest / trade / roll")
    # Modes
    parser.add_argument("--trade", action="store_true", help="Place idempotent paper orders for current signals")
    parser.add_argument("--roll", action="store_true", help="Cancel/replace bracket orders after US close")
    parser.add_argument("--backtest", action="store_true", help="Run historical backtest (downloads daily bars)")
    # Connection
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7497, help="7497 paper, 7496 live")
    parser.add_argument("--client-id", type=int, default=44)
    # Symbols
    parser.add_argument("--symbols", nargs="*", help="Space-separated symbols. If omitted, uses the default list.")
    parser.add_argument("--symbols-file", help="Path to a text file with one symbol per line")
    # Strategy selection + overrides
    parser.add_argument("--strategy", default="adx_squeeze", help="Strategy key (see registry)")
    parser.add_argument("--len-channel", type=int, default=None)
    parser.add_argument("--adx-len", type=int, default=None)
    parser.add_argument("--adx-thresh", type=float, default=None)
    parser.add_argument("--trade-pct", type=float, default=None)
    parser.add_argument("--max-positions", type=int, default=None)
    parser.add_argument("--max-exposure-pct", type=float, default=None)
    parser.add_argument("--warmup-bars", type=int, default=None)
    parser.add_argument("--rsi-len", type=int, default=None)
    parser.add_argument("--rsi-thresh", type=float, default=None)

    # Utilities
    parser.add_argument("--force-roll", action="store_true", help="Run roll even if not after US close")
    parser.add_argument("--preview", action="store_true", help="Preview actions without placing orders (trade/roll)")
    parser.add_argument("--outdir", default="output", help="Backtest output folder")
    parser.add_argument("--price-tol", type=float, default=None, help="Treat stops equal within this absolute amount")
    args = parser.parse_args()
    
    if args.price_tol is not None:
        PRICE_TOL = float(args.price_tol)
    # Symbols (CLI overrides default)
    if args.symbols_file:
        with open(args.symbols_file, "r", encoding="utf-8") as fh:
            symbols = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
    elif args.symbols:
        symbols = args.symbols
    else:
        symbols = ["MSFT", "NVDA", "AAPL", "AMZN", "GOOGL", "META", "AVGO", "TSLA", "WMT", "JPM", "V", "SPY", "BRK.A"]
    # Strategy instantiation (CLI overrides fall back to your constants)
    overrides = {}
    if args.len_channel is not None: overrides["len_channel"] = args.len_channel
    if args.adx_len is not None: overrides["adx_len"] = args.adx_len
    if args.adx_thresh is not None: overrides["adx_thresh"] = args.adx_thresh
    if args.rsi_len is not None: overrides["rsi_len"] = args.rsi_len
    if args.rsi_thresh is not None: overrides["rsi_thresh"] = args.rsi_thresh

    if args.trade_pct is not None: overrides["trade_pct"] = args.trade_pct
    if args.max_positions is not None: overrides["max_positions"] = args.max_positions
    if args.max_exposure_pct is not None: overrides["max_exposure_pct"] = args.max_exposure_pct
    if args.warmup_bars is not None: overrides["warmup_bars"] = args.warmup_bars
    overrides.setdefault("len_channel", LEN)
    overrides.setdefault("adx_len", ADXLEN)
    overrides.setdefault("adx_thresh", ADXTHRESH)
    overrides.setdefault("trade_pct", TRADE_PCT)
    overrides.setdefault("max_positions", MAX_POSITIONS)
    overrides.setdefault("max_exposure_pct", MAX_EXPOSURE_PCT)
    overrides.setdefault("warmup_bars", WARMUP_BARS)
    strat = build_strategy(args.strategy, **overrides)
    # --- Mode routing ---
    if args.backtest:
        usable = {}
        app = IbApp()
        try:
            app.connect(args.host, args.port, clientId=args.client_id)
            import threading
            reader = threading.Thread(target=app.run, daemon=True)
            reader.start()
            if not app.connected_evt.wait(10):
                raise TimeoutError("Failed to connect to IB for historical download")
            for sym in symbols:
                df = latest_completed_daily_df(app, sym, strategy=strat)
                if not df.empty:
                    usable[sym] = df
        finally:
            if app.isConnected():
                app.disconnect()
        trades_df, summary, equity_df = backtest_portfolio(usable, strategy=strat)
        os.makedirs(args.outdir, exist_ok=True)
        save_csvs(args.outdir, trades_df, summary, equity_df)
        print(f"[BACKTEST] Saved CSVs to: {args.outdir}")
    elif args.roll:
        if args.force_roll:
            def _ok(): return True
            globals()["_after_us_close_now"] = _ok
        if args.preview:
            print("[PREVIEW] Roll would run now (no orders placed).")
        else:
            roll_daily_brackets_after_close(symbols, strategy=strat, host=args.host, port=args.port, client_id=args.client_id)
    else:
        if args.preview:
            print("[PREVIEW] Would place paper orders now (no orders placed).")
        else:
            place_paper_orders_now(symbols, strategy=strat, host=args.host, port=args.port, client_id=args.client_id)
