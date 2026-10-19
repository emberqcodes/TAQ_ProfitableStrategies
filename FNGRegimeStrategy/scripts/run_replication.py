"""
run_replication.py -- press-play: rebuild the regime, verify vs the manifest, and
reproduce the validated ETH execution result (Sharpe ~1.12). Run from this folder:
    python run_replication.py
Expects Data/ populated (the parquets ship with this package; re-pull to refresh).
"""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np, pandas as pd
from config import PRICE, FUNDING, SUPPLY, FEARGREED, MANIFEST
import regime as R
import execution_engine as E

SUPPLY_FILE = SUPPLY / "full_basket_supply.parquet"      # <-- 3-coin basket (critical)
FNG_FILE    = FEARGREED / "alternative_me_fng.parquet"
CPT = 0.001

def daily_ref_sharpe(states, close, fund, lag):
    perm = states.reindex(close.index).shift(lag).fillna(0).astype(float)
    r = close.pct_change().fillna(0.0)
    turn = CPT * perm.diff().abs().fillna(perm.abs()); fc = perm * fund.reindex(close.index).fillna(0)
    d = (perm * r - turn - fc); d = d[(perm != 0).cumsum().astype(bool)]
    return d.mean() / d.std() * np.sqrt(365)

def manifest_agreement(states, asset):
    m = json.load(open(MANIFEST))
    s = pd.DataFrame(m["regime"]["states"][asset], columns=["date", "state"]); s["date"] = pd.to_datetime(s["date"])
    ms = s.set_index("date")["state"].astype(int)
    both = pd.concat([states.rename("r"), ms.rename("m")], axis=1).dropna()
    return 100 * (both["r"] == both["m"]).mean()

print("="*70); print("REGIME REBUILD + VERIFY (Option A, 3-coin basket)"); print("="*70)
for tag, sym in [("ETH", "ETHUSDT"), ("BTC", "BTCUSDT")]:
    pf, ff = PRICE / f"binance_{sym}_1d.parquet", FUNDING / f"binance_{sym}.parquet"
    if not pf.exists(): print(f"  {tag}: price file missing, skipping"); continue
    pan = R.build_panel(pf, ff, SUPPLY_FILE, FNG_FILE, 365)
    states = R.run_stateless(pan)
    close = R.load_price_daily(pf); fund = R.load_funding_daily(ff)
    agree = manifest_agreement(states, tag.lower())
    s1, s2 = daily_ref_sharpe(states, close, fund, 1), daily_ref_sharpe(states, close, fund, 2)
    print(f"  {tag}: manifest agreement {agree:.0f}% (want >=95%)   daily Sharpe T+1={s1:.2f} T+2={s2:.2f}")

print("\n" + "="*70); print("ETH EXECUTION LAYER (Coinbase 1h, ATR14x15 trailing, T+2, funding on)"); print("="*70)
cb = PRICE / "coinbase_ETHUSD_1h.parquet"
if cb.exists():
    pan = R.build_panel(PRICE/"binance_ETHUSDT_1d.parquet", FUNDING/"binance_ETHUSDT.parquet", SUPPLY_FILE, FNG_FILE, 365)
    c = pd.read_parquet(cb); c["ts"] = pd.to_datetime(c["open_time"], unit="ms", utc=True).dt.tz_localize(None)
    bars = c.set_index("ts")[["open","high","low","close"]].sort_index(); bars = bars[bars.index >= "2020-11-25"]
    f = pd.read_parquet(FUNDING/"binance_ETHUSDT.parquet"); f["ts"] = pd.to_datetime(f["funding_time"], unit="ms", utc=True).dt.tz_localize(None)
    fund8h = f.set_index("ts")["funding_rate"].astype(float).sort_index()
    for label, st in [("stateless", R.run_stateless(pan)), ("hysteresis d=5", R.run_hyst(pan, delta=5))]:
        sti = st.copy(); sti.index = sti.index.normalize()
        led, s = E.run_execution(bars, sti, funding_8h=fund8h, atr_n=14, trail=15.0, risk=2000.0, base=30000.0, max_lev=25.0, lag_days=2)
        print(f"  {label:<16} Sharpe {s['sharpe']:.2f}   return {s['ret_pct']:.0f}%   maxDD {s['max_dd_pct']:.0f}%   long ${s['long_pnl']:,.0f} / short ${s['short_pnl']:,.0f}")
    print("\n  Expected: stateless ~1.12 / -23% ; hysteresis d=5 ~1.17 / -19%")
else:
    print("  coinbase_ETHUSD_1h.parquet missing -- pull it to run the execution layer.")
