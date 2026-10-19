"""
04_fng_trustcheck.py  --  ETH Fear & Greed source vetting (NOT ingestion).

Decides whether an ETH-specific Fear & Greed source is trustworthy enough to feed
the regime. Three checks:

  1) ACCESSIBLE  -- pulls without payment / key?            (immediate)
  2) DEPTH       -- reaches >= NEED_START so the trailing-   (immediate)
                    365 percentile is warm by BACKTEST_START
  3) RESTATEMENT -- do already-settled historical values     (needs 2 snapshots
                    change between pulls? (the killer)         taken days apart)

Modes:
  python 04_fng_trustcheck.py snapshot <source>   # pull + dated save + checks 1&2
  python 04_fng_trustcheck.py compare  <source>   # diff two snapshots -> check 3

Sources:
  alternative_me  -- clean control + your current crypto-wide placeholder (free)
  cfgi_eth        -- the genuine ETH-specific candidate (gated API; history since
                     2021 only; AI methodology -> restatement-prone). Endpoint/auth
                     NOT wired: fill in fetch_cfgi_eth() once you have access.

Snapshots: Data/fear_greed/snapshots/{source}_{YYYY-MM-DD}.parquet
NOTE: sources unreachable from the build environment -> diff/depth logic is
self-tested (04_fng_trustcheck_selftest.py); run live locally.
"""
import sys
import requests
import pandas as pd
from config import FEARGREED, BACKTEST_START

WARMUP_DAYS = 365
NEED_START = pd.Timestamp(BACKTEST_START) - pd.Timedelta(days=WARMUP_DAYS)
SETTLE_SKIP_DAYS = 3        # the most recent few days legitimately still update
SNAP_DIR = FEARGREED / "snapshots"

# ============================== source adapters ===============================
# Each returns DataFrame[date, value] (value 0-100), deduped, sorted.
def fetch_alternative_me():
    r = requests.get("https://api.alternative.me/fng/",
                     params={"limit": 0, "format": "json"}, timeout=30)
    r.raise_for_status()
    data = r.json()["data"]
    df = pd.DataFrame(
        [(pd.Timestamp(int(d["timestamp"]), unit="s").normalize(), int(d["value"])) for d in data],
        columns=["date", "value"])
    return df.drop_duplicates("date").sort_values("date").reset_index(drop=True)

def fetch_cfgi_eth():
    # CFGI developer API is gated and its history starts 2021. Wire the endpoint +
    # auth here once you have access; parse into DataFrame[date, value].
    raise NotImplementedError(
        "CFGI endpoint/auth not wired. Docs: history since 2021 only, gated access, "
        "AI methodology (restatement risk). Likely fails DEPTH for a 2020-10 backtest.")

SOURCES = {"alternative_me": fetch_alternative_me, "cfgi_eth": fetch_cfgi_eth}
REFERENCE = "alternative_me"

# ============================== pure helpers (tested) =========================
def depth_ok(earliest, need_start=NEED_START):
    return pd.Timestamp(earliest) <= pd.Timestamp(need_start)

def restatement_report(old, new, tail_skip_days=SETTLE_SKIP_DAYS):
    """Diff two snapshots on SETTLED overlapping dates. Any change == restatement."""
    m = old.merge(new, on="date", suffixes=("_old", "_new"))
    if len(m) == 0:
        return {"n_overlap": 0, "n_settled": 0, "n_changed": 0, "max_drift": 0.0, "worst": []}
    cutoff = old["date"].max() - pd.Timedelta(days=tail_skip_days)
    settled = m[m["date"] <= cutoff].copy()
    settled["drift"] = settled["value_new"] - settled["value_old"]
    changed = settled[settled["drift"] != 0]
    worst = (changed.reindex(changed["drift"].abs().sort_values(ascending=False).index)
             .head(5)[["date", "value_old", "value_new"]].values.tolist())
    return {"n_overlap": len(m), "n_settled": len(settled), "n_changed": len(changed),
            "max_drift": float(changed["drift"].abs().max()) if len(changed) else 0.0,
            "worst": worst}

def overlap_corr(a, b):
    """Pearson corr of two [date,value] series on overlapping dates (is it ETH-distinct?)."""
    m = a.merge(b, on="date", suffixes=("_a", "_b"))
    return float(m["value_a"].corr(m["value_b"])) if len(m) > 2 else float("nan")

# ============================== modes =========================================
def _snap_path(source, day):
    return SNAP_DIR / f"{source}_{day}.parquet"

def _latest_snaps(source, k=2):
    files = sorted(SNAP_DIR.glob(f"{source}_*.parquet"))
    return files[-k:]

def snapshot(source):
    df = SOURCES[source]()
    SNAP_DIR.mkdir(parents=True, exist_ok=True)
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    path = _snap_path(source, today)
    df.to_parquet(path, index=False)
    earliest, latest = df["date"].min().date(), df["date"].max().date()
    print(f"[{source}] snapshot {today}: {len(df)} rows, {earliest} -> {latest}")
    print("  check 1 ACCESSIBLE: ok (pulled without payment)")
    ok = depth_ok(earliest)
    print(f"  check 2 DEPTH: {'ok' if ok else 'FAIL'} "
          f"(need <= {NEED_START.date()} for 365d warmup; have {earliest})")
    # is it actually ETH-distinct vs the crypto-wide reference?
    if source != REFERENCE:
        refs = _latest_snaps(REFERENCE, 1)
        if refs:
            c = overlap_corr(df, pd.read_parquet(refs[-1]))
            print(f"  note: corr vs {REFERENCE} = {c:.3f} "
                  f"({'barely ETH-distinct' if c > 0.95 else 'meaningfully distinct'})")
        else:
            print(f"  note: snapshot {REFERENCE} too, to measure ETH-distinctness.")
    print("  check 3 RESTATEMENT: needs a second snapshot days from now, then `compare`.")

def compare(source):
    snaps = _latest_snaps(source, 2)
    if len(snaps) < 2:
        print(f"[{source}] need 2 snapshots to compare; have {len(snaps)}. "
              f"Run `snapshot {source}` again after a few days.")
        return
    old, new = pd.read_parquet(snaps[0]), pd.read_parquet(snaps[1])
    print(f"[{source}] compare {snaps[0].stem.split('_')[-1]} -> {snaps[1].stem.split('_')[-1]}")
    rep = restatement_report(old, new)
    print(f"  settled overlap: {rep['n_settled']} dates")
    if rep["n_changed"] == 0:
        print("  check 3 RESTATEMENT: ok -- no settled value changed (point-in-time clean)")
    else:
        print(f"  check 3 RESTATEMENT: FAIL -- {rep['n_changed']} settled dates changed, "
              f"max drift {rep['max_drift']:.0f}")
        for d, a, b in rep["worst"]:
            print(f"      {pd.Timestamp(d).date()}: {a} -> {b}")

if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("snapshot", "compare") or sys.argv[2] not in SOURCES:
        print("usage: python 04_fng_trustcheck.py {snapshot|compare} {" + "|".join(SOURCES) + "}")
        sys.exit(1)
    (snapshot if sys.argv[1] == "snapshot" else compare)(sys.argv[2])
