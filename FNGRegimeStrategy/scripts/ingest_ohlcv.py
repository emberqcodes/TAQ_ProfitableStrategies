"""
01_ohlcv.py  --  ETH OHLCV ingest across venues (supersedes 01_perp_ohlcv.py).

Venues:
  binance   ETHUSDT, ETHUSDC          PERP (USD-M futures)   forward-paged
  bybit     ETHUSDT, ETHUSDC(=ETHPERP) PERP (linear)          forward-paged
  coinbase  ETHUSD, ETHUSDT, ETHUSDC   SPOT                    backward-paged (300/call)

Multi-timeframe: INTERVALS is a list; each venue pulls the ones it supports and
skips the rest (e.g. Coinbase has no 4h). All venues normalise to one schema:
    [open_time(ms,UTC), open, high, low, close, volume]
so resume + merge are identical across venues.

  Output:  Data/price/{venue}_{canonical}_{interval}.parquet

RESUMABLE + IDEMPOTENT. Venues unreachable from the build environment -> the
pure/normalisation/paging logic is self-tested (01_ohlcv_selftest.py); run live
locally. NOTE: Coinbase max history differs per product (ETH-USD is deep, the
USDT/USDC spot pairs are younger) and fine granularities (1m/5m) are SLOW on
Coinbase because of the 300-candle-per-request cap.
"""
import time
import requests
import pandas as pd
from config import PRICE, CONTRACTS

# ----------------------------- config knobs -----------------------------------
INTERVALS = ["1h", "1d"]        # any of: 1m 5m 15m 1h 4h 6h 1d  (per-venue support below)
INCEPTION_MS = 1546300800000    # 2019-01-01 UTC; forward-paged venues start here on a fresh pull
REQUEST_PAUSE = 0.25            # seconds between calls

COLS = ["open_time", "open", "high", "low", "close", "volume"]
UA = {"User-Agent": "eth-regime-ingest/1.0"}

# canonical interval -> venue-specific code. Missing key = venue can't do it (skipped).
VENUE_INTERVALS = {
    "binance":  {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "6h": "6h", "1d": "1d"},
    "bybit":    {"1m": "1", "5m": "5", "15m": "15", "1h": "60", "4h": "240", "6h": "360", "1d": "D"},
    "coinbase": {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "6h": 21600, "1d": 86400},  # no 4h
}
INTERVAL_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
               "4h": 14_400_000, "6h": 21_600_000, "1d": 86_400_000}

# --------------------------- pure helpers (tested) -----------------------------
def now_ms():
    return int(time.time() * 1000)

def last_saved_ms(path):
    """Max open_time already on disk, or None if no file yet."""
    if path.exists():
        df = pd.read_parquet(path, columns=["open_time"])
        if len(df):
            return int(df["open_time"].max())
    return None

def merge_save(path, new_rows):
    new = pd.DataFrame(new_rows, columns=COLS)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    out = new.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(out)

def _iso(ms):
    return pd.Timestamp(ms, unit="ms", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ")

# ------------------------------- venue fetchers --------------------------------
# Each yields oldest-first pages of normalised [open_time_ms, o, h, l, c, v].
def fetch_binance(symbol, interval, last_ms):
    url = "https://fapi.binance.com/fapi/v1/klines"
    iv, step, limit = VENUE_INTERVALS["binance"][interval], INTERVAL_MS[interval], 1500
    start = (last_ms + 1) if last_ms is not None else INCEPTION_MS
    while True:
        r = requests.get(url, params={"symbol": symbol, "interval": iv,
                                      "startTime": start, "limit": limit}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return
        yield [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in data]
        if len(data) < limit:
            return
        start = int(data[-1][0]) + step
        time.sleep(REQUEST_PAUSE)

def fetch_bybit(symbol, interval, last_ms):
    url = "https://api.bybit.com/v5/market/kline"
    iv, step, limit = VENUE_INTERVALS["bybit"][interval], INTERVAL_MS[interval], 1000
    start = (last_ms + 1) if last_ms is not None else INCEPTION_MS
    while True:
        r = requests.get(url, params={"category": "linear", "symbol": symbol,
                                      "interval": iv, "start": start, "limit": limit}, timeout=20)
        r.raise_for_status()
        lst = r.json().get("result", {}).get("list", [])
        if not lst:
            return
        lst = sorted(lst, key=lambda k: int(k[0]))
        yield [[int(k[0]), float(k[1]), float(k[2]), float(k[3]), float(k[4]), float(k[5])] for k in lst]
        if len(lst) < limit:
            return
        start = int(lst[-1][0]) + step
        time.sleep(REQUEST_PAUSE)

def fetch_coinbase(symbol, interval, last_ms):
    """Backward-page from now in 300-candle windows; stop at existing data or listing.
    Coinbase candle = [time(sec), low, high, open, close, volume] -> normalise + reorder."""
    gran = VENUE_INTERVALS["coinbase"][interval]
    step_ms = INTERVAL_MS[interval]
    window = 300 * step_ms
    floor_ms = last_ms if last_ms is not None else -1
    end = now_ms()
    prev_oldest = None
    url = f"https://api.exchange.coinbase.com/products/{symbol}/candles"
    while True:
        start = end - window
        r = requests.get(url, params={"granularity": gran, "start": _iso(start), "end": _iso(end)},
                         headers=UA, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return                                  # before listing / no ticks
        rows = [[int(c[0]) * 1000, float(c[3]), float(c[2]), float(c[1]), float(c[4]), float(c[5])]
                for c in data]                      # [t*1000, open, high, low, close, vol]
        yield rows
        oldest = min(r_[0] for r_ in rows)
        if oldest <= floor_ms:
            return                                  # caught up to what we already have
        if prev_oldest is not None and oldest >= prev_oldest:
            return                                  # no progress -> hit the listing boundary, stop
        prev_oldest = oldest
        end = oldest - 1                            # exclude boundary candle (Coinbase end is inclusive)
        time.sleep(REQUEST_PAUSE)

FETCHERS = {"binance": fetch_binance, "bybit": fetch_bybit, "coinbase": fetch_coinbase}

# --------------------------------- driver --------------------------------------
def ingest(venue, canonical, interval):
    symbol = CONTRACTS[venue][canonical]
    path = PRICE / f"{venue}_{canonical}_{interval}.parquet"
    last = last_saved_ms(path)
    resume = "fresh" if last is None else f"resume>{pd.to_datetime(last, unit='ms', utc=True)}"
    print(f"[{venue} {canonical} ({symbol}) {interval}] {resume}")
    total, pages = 0, 0
    for rows in FETCHERS[venue](symbol, interval, last):
        total = merge_save(path, rows)
        pages += 1
        edge = pd.to_datetime(min(r_[0] for r_ in rows), unit="ms", utc=True)
        print(f"    page {pages}: +{len(rows)} -> {total} rows (oldest this page {edge})")
    if path.exists():
        df = pd.read_parquet(path, columns=["open_time"])
        if len(df):
            lo = pd.to_datetime(df["open_time"].min(), unit="ms", utc=True).date()
            hi = pd.to_datetime(df["open_time"].max(), unit="ms", utc=True).date()
            print(f"    coverage: {lo} -> {hi}  ({len(df)} bars)  {path.name}")

if __name__ == "__main__":
    print(f"ETH OHLCV ingest -- intervals={INTERVALS}")
    for venue in CONTRACTS:
        for canonical in CONTRACTS[venue]:
            for interval in INTERVALS:
                if interval not in VENUE_INTERVALS[venue]:
                    print(f"[{venue} {canonical} {interval}] skip (unsupported on {venue})")
                    continue
                try:
                    ingest(venue, canonical, interval)
                except Exception as e:
                    print(f"    !! {venue} {canonical} {interval} FAILED: {e}")
    print("all done.")
