"""
05_fng.py  --  Fear & Greed ingest (alternative.me, crypto-wide).

The crypto-wide F&G the BTC regime already uses -- kept for the ETH regime since
no ETH-specific source is free + deep + clean (see 04_fng_trustcheck.py). It's the
lowest-weight signal anyway (shares the risk-cycle axis with funding).

alternative.me returns FULL daily history in one call (limit=0), so this just
pulls everything and overwrites -- no paging, no resume bookkeeping needed. Past
values are point-in-time clean; only today's value updates intraday, so re-running
simply refreshes it.

  Output:  Data/fear_greed/alternative_me_fng.parquet   (date, value 0-100, classification)

NOTE: api.alternative.me is unreachable from the build environment -> the parse
logic is self-tested (05_fng_selftest.py); run live locally.
"""
import requests
import pandas as pd
from config import FEARGREED

def fetch():
    r = requests.get("https://api.alternative.me/fng/",
                     params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    rows = [(pd.Timestamp(int(d["timestamp"]), unit="s").normalize(),
             int(d["value"]), d.get("value_classification", "")) for d in data]
    df = pd.DataFrame(rows, columns=["date", "value", "classification"])
    return df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)

def run():
    df = fetch()
    FEARGREED.mkdir(parents=True, exist_ok=True)
    path = FEARGREED / "alternative_me_fng.parquet"
    df.to_parquet(path, index=False)
    lo, hi = df["date"].min().date(), df["date"].max().date()
    print(f"[alternative.me F&G] {len(df)} rows, {lo} -> {hi}  ({path.name})")
    print(f"    latest: {df['value'].iloc[-1]} ({df['classification'].iloc[-1]})")

if __name__ == "__main__":
    run()
