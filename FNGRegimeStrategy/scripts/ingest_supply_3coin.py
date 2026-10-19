"""
ingest_supply_3coin.py -- CRITICAL: the 3-coin stablecoin basket (Coin Metrics).

Basket = usdt_ethereum + usdt_tron + usdc_ethereum. DROPPING USDT-TRON is the bug that
degrades the regime from ~1.0 to ~0.5. Writes Data/supply/full_basket_supply.parquet.
Run locally in the .venv (community-api.coinmetrics.io is not reachable from a sandbox).
Confirm the probe says 'ok' for all three codes before trusting the pull.
"""
import numpy as np, pandas as pd, requests
from config import SUPPLY

BASE = "https://community-api.coinmetrics.io/v4"
METRIC = "SplyCur"; ROC_DAYS = 28; START = "2017-01-01"
BASKET_CODES = ["usdt_eth", "usdt_trx", "usdc"]      # usdt_eth + USDT-Tron + usdc(Ethereum)
PROBE_EXTRA  = ["usdt_tron", "usdc_eth"]             # fallbacks if a code doesn't resolve
OUT = SUPPLY / "full_basket_supply.parquet"

def rows_to_series(data, metric):
    out = []
    for d in data:
        v = d.get(metric)
        out.append([pd.Timestamp(d["time"]).tz_convert("UTC").normalize().tz_localize(None),
                    np.nan if v is None else float(v)])
    return pd.DataFrame(out, columns=["date","supply"]).drop_duplicates("date").sort_values("date").reset_index(drop=True)

def align_sum(members):
    wide = None
    for code, df in members.items():
        s = df.set_index("date")["supply"].rename(code)
        wide = s.to_frame() if wide is None else wide.join(s, how="outer")
    wide = wide.sort_index()
    return wide.sum(axis=1, skipna=False).rename("total"), wide   # NaN if any member missing

def roc(series, n=ROC_DAYS): return (series/series.shift(n) - 1.0).rename(f"roc_{n}d")

def probe(code):
    r = requests.get(f"{BASE}/timeseries/asset-metrics",
                     params={"assets":code,"metrics":METRIC,"frequency":"1d","limit_per_asset":1,"page_size":1}, timeout=30)
    if r.status_code != 200: return f"HTTP {r.status_code}"
    data = r.json().get("data", [])
    if not data: return "no data (not community / wrong code)"
    return f"ok   latest {data[0]['time'][:10]} = {float(data[0][METRIC]):,.0f}"

def fetch_supply(code, start=START):
    url = f"{BASE}/timeseries/asset-metrics"
    params = {"assets":code,"metrics":METRIC,"frequency":"1d","start_time":start,"page_size":10000}
    frames = []
    while url:
        r = requests.get(url, params=params, timeout=60); r.raise_for_status(); j = r.json()
        frames.append(rows_to_series(j.get("data", []), METRIC)); url = j.get("next_page_url"); params = None
    if not frames: return pd.DataFrame(columns=["date","supply"])
    return pd.concat(frames, ignore_index=True).drop_duplicates("date").sort_values("date").reset_index(drop=True)

def run():
    print("== probe ==")
    for code in BASKET_CODES + PROBE_EXTRA: print(f"  {code:11s} {probe(code)}")
    bad = [c for c in BASKET_CODES if not probe(c).startswith("ok")]
    if bad: print(f"\n!! did not resolve: {bad} -- fix BASKET_CODES (e.g. usdt_trx->usdt_tron) and re-run."); return
    members = {}
    for code in BASKET_CODES:
        df = fetch_supply(code); SUPPLY.mkdir(parents=True, exist_ok=True)
        df.to_parquet(SUPPLY / f"raw_{code}.parquet", index=False); members[code] = df
        print(f"  pulled {code:11s} {len(df)} days")
    total, wide = align_sum(members); valid = total.dropna()
    out = pd.DataFrame({"date": total.index, "total": total.values}); out["roc_28d"] = roc(total).values
    out = out.dropna(subset=["total"]).reset_index(drop=True); out.to_parquet(OUT, index=False)
    print(f"  wrote {OUT.name}: {len(out)} rows, basket valid from {valid.index.min().date()}")

if __name__ == "__main__":
    run()
