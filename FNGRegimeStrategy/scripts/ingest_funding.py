"""
02_funding.py  --  ETH perpetual funding-rate ingest (Binance + Bybit).

Tiny download (~3 stamps/day per contract). RESUMABLE + IDEMPOTENT, mirrors 01.

  Output:  Data/funding/{venue}_{canonical}.parquet
           columns: funding_time(ms, UTC), funding_rate

NOTE: venues unreachable from the build environment -> logic is self-tested
(02_funding_selftest.py); run the real pull locally. Bybit funding history can
cap how far back it serves; the coverage line at the end shows what you got.
The old BTC pipeline's startTime=0 bug is avoided here: we page forward from an
explicit inception, never from zero.
"""
import time
import requests
import pandas as pd
from config import FUNDING, CONTRACTS

# ----------------------------- config knobs -----------------------------------
INCEPTION_MS = 1546300800000    # 2019-01-01 UTC; venues return from each contract's listing
REQUEST_PAUSE = 0.25

COLS = ["funding_time", "funding_rate"]

# --------------------------- pure helpers (tested) -----------------------------
def resume_start_ms(path, default_ms):
    if path.exists():
        df = pd.read_parquet(path, columns=["funding_time"])
        if len(df):
            return int(df["funding_time"].max()) + 1
    return default_ms

def merge_save(path, new_rows):
    new = pd.DataFrame(new_rows, columns=COLS)
    if path.exists():
        new = pd.concat([pd.read_parquet(path), new], ignore_index=True)
    out = new.drop_duplicates("funding_time").sort_values("funding_time").reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(path, index=False)
    return len(out)

# ------------------------------- venue fetchers --------------------------------
def fetch_binance(symbol, start_ms):
    """Yield oldest-first pages of [funding_time, funding_rate] from Binance USD-M."""
    url = "https://fapi.binance.com/fapi/v1/fundingRate"
    limit = 1000
    while True:
        r = requests.get(url, params={"symbol": symbol, "startTime": start_ms, "limit": limit}, timeout=20)
        r.raise_for_status()
        data = r.json()
        if not data:
            return
        yield [[int(x["fundingTime"]), float(x["fundingRate"])] for x in data]
        if len(data) < limit:
            return
        start_ms = int(data[-1]["fundingTime"]) + 1
        time.sleep(REQUEST_PAUSE)

def fetch_bybit(symbol, start_ms):
    """Yield oldest-first pages from Bybit v5 linear funding. (Bybit is newest-first; reversed here.)"""
    url = "https://api.bybit.com/v5/market/funding/history"
    limit = 200
    while True:
        r = requests.get(url, params={"category": "linear", "symbol": symbol,
                                      "startTime": start_ms, "limit": limit}, timeout=20)
        r.raise_for_status()
        lst = r.json().get("result", {}).get("list", [])
        if not lst:
            return
        lst = sorted(lst, key=lambda x: int(x["fundingRateTimestamp"]))
        yield [[int(x["fundingRateTimestamp"]), float(x["fundingRate"])] for x in lst]
        if len(lst) < limit:
            return
        start_ms = int(lst[-1]["fundingRateTimestamp"]) + 1
        time.sleep(REQUEST_PAUSE)

FETCHERS = {"binance": fetch_binance, "bybit": fetch_bybit}

# --------------------------------- driver --------------------------------------
def ingest(venue, canonical):
    symbol = CONTRACTS[venue][canonical]
    path = FUNDING / f"{venue}_{canonical}.parquet"
    start = resume_start_ms(path, INCEPTION_MS)
    print(f"[{venue} {canonical} ({symbol})] resume from {pd.to_datetime(start, unit='ms', utc=True)}")
    total, pages = 0, 0
    for rows in FETCHERS[venue](symbol, start):
        total = merge_save(path, rows)
        pages += 1
        through = pd.to_datetime(rows[-1][0], unit="ms", utc=True)
        print(f"    page {pages}: +{len(rows)} -> {total} stamps (through {through})")
    print(f"    done: {total} stamps in {path.name}")
    if path.exists():
        df = pd.read_parquet(path, columns=["funding_time"])
        if len(df):
            lo = pd.to_datetime(df["funding_time"].min(), unit="ms", utc=True)
            hi = pd.to_datetime(df["funding_time"].max(), unit="ms", utc=True)
            print(f"    coverage: {lo.date()} -> {hi.date()}  ({len(df)} stamps)")

if __name__ == "__main__":
    print("ETH perp funding ingest (Binance only; Bybit funding added manually later)")
    for venue in ("binance",):
        for canonical in ("ETHUSDT", "ETHUSDC"):
            try:
                ingest(venue, canonical)
            except Exception as e:
                print(f"    !! {venue} {canonical} FAILED: {e}")
    print("all done.")
