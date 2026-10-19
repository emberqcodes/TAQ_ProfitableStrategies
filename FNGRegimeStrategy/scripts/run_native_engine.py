"""
run_native_engine.py -- run the company's native two-file engine
(regime_backtest_engine.py + strategy_manifest.json) on the bundled data.
Reads ONLY the manifest's embedded states (no signal recompute) -- the platform path.
Independent cross-check of run_replication.py (the clean reconstruction).
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from config import PRICE, FUNDING, MANIFEST
import regime_backtest_engine as NE

def bars(path):
    d = pd.read_parquet(path); d["ts"] = pd.to_datetime(d["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    d = d.set_index("ts")[["open","high","low","close"]].sort_index()
    return d[d.index >= "2020-11-25"]

def fund_arr(path, idx):
    f = pd.read_parquet(path); f["ts"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True).dt.tz_localize(None)
    return f.set_index("ts")["funding_rate"].astype(float).sort_index().reindex(idx).fillna(0.0).values

m = NE.load_manifest(str(MANIFEST))
eth = bars(PRICE/"coinbase_ETHUSD_1h.parquet")          # ETH executes on Coinbase (trade venue)
btc = bars(PRICE/"binance_BTCUSDT_1h.parquet")          # BTC on Binance 1h
ohlcv = {"btc": btc, "eth": eth}
funding = {"eth": fund_arr(FUNDING/"binance_ETHUSDT.parquet", eth.index),
           "btc": fund_arr(FUNDING/"binance_BTCUSDT.parquet", btc.index)}
print("Native engine -- ETH=Coinbase 1h, BTC=Binance 1h, embedded states, T+2")
print("(Sharpe here is per-bar-annualized; run_replication.py reports daily-resampled.)\n")
NE.run_backtest(m, ohlcv, funding, verbose=True)
