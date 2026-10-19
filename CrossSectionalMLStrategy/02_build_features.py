"""
Step 11 - Vectorised feature build for every symbol in data_processed/xs/build_symbols.txt.

Same definitions as steps 2-4 (bar label = open time, known_at_ts = open + 15m, intra-bar
features from the 1m rows inside the bar, funding visible only if settled strictly before
known_at_ts, label = 1m open at known_at_ts + h minus 1m open at known_at_ts, over ATR14),
but the intra-bar features are computed with groupby aggregates instead of a per-bar apply,
so a full-history symbol takes seconds rather than minutes.

Per symbol -> data_processed/xs/feat/<SYM>.parquet   (float32 features, one row per 15m bar)
             data_processed/xs/slim/<SYM>.parquet    (1m open/high/low for fills later)
             data_processed/xs/fund/<SYM>.parquet    (clean funding events)
Resumable: symbols with an existing feature file are skipped. Delete the file to rebuild.

Lookahead self-test: the truncation test from step 4 is run on the first
LOOKAHEAD_TEST_SYMBOLS symbols built.
"""
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # Windows console safety
from common import load_ohlcv_1m, load_funding

# import compute_bar_features and BAR_FEATURES from features_lib.py (module name starts with a digit)
_spec = importlib.util.spec_from_file_location("f04", C.PROJECT_DIR / "features_lib.py")
f04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(f04)

BAR = f"{C.BAR_MINUTES}min"
EPS = 1e-12
H = C.XS_HORIZON
LOOKAHEAD_TEST_SYMBOLS = 3
FEAT_DIR, SLIM_DIR, FUND_DIR = C.XS_DIR / "feat", C.XS_DIR / "slim", C.XS_DIR / "fund"
for d in (FEAT_DIR, SLIM_DIR, FUND_DIR):
    d.mkdir(exist_ok=True)


def build_15m(df: pd.DataFrame) -> pd.DataFrame:
    """vectorised 15m bars + intra-bar features from 1m rows."""
    df = df.copy()
    df["bar_open_ts"] = df["timestamp"].dt.floor(BAR)
    df["min_in_bar"] = ((df["timestamp"] - df["bar_open_ts"]).dt.total_seconds() // 60).astype(int)
    g = df.groupby("bar_open_ts", sort=True)
    lr = np.log(df["close"]).groupby(df["bar_open_ts"]).diff()          # 1m log return within bar (NaN first)
    df["lr"] = lr
    df["abs_lr"] = lr.abs()
    df["lr2"] = lr ** 2
    last5 = df["min_in_bar"] >= C.BAR_MINUTES - 5
    df["tbv5"] = df["taker_buy_volume"].where(last5, 0.0)
    df["vol5"] = df["volume"].where(last5, 0.0)
    df["up"] = (df["close"] > df["open"]).astype(float)

    agg = g.agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"),
                volume=("volume", "sum"), quote_volume=("quote_volume", "sum"), trades=("trades", "sum"),
                taker_buy_volume=("taker_buy_volume", "sum"), taker_buy_quote_volume=("taker_buy_quote_volume", "sum"),
                n_min=("open", "size"), path_len=("abs_lr", "sum"), tbv5=("tbv5", "sum"), vol5=("vol5", "sum"),
                up_frac=("up", "mean"), lr_mean=("lr", "mean"), lr2_mean=("lr2", "mean"), ret_skew=("lr", "skew"))
    b = agg.reset_index()
    b["ib_taker_buy_ratio"] = b["taker_buy_volume"] / (b["volume"] + EPS)
    b["ib_taker_buy_ratio_last5"] = b["tbv5"] / (b["vol5"] + EPS)
    b["ib_up_frac"] = b["up_frac"]
    b["ib_path_len"] = b["path_len"]
    b["ib_net_move"] = np.abs(np.log(b["close"] / b["open"]))
    b["ib_ret_skew"] = b["ret_skew"]
    b["ib_vol_last5_share"] = b["vol5"] / (b["volume"] + EPS)
    b["ib_ret_std_1m"] = np.sqrt(np.clip(b["lr2_mean"] - b["lr_mean"] ** 2, 0, None))   # population std, as in step 2
    b["ib_chop"] = b["ib_path_len"] / (b["ib_net_move"] + 1e-6)
    b = b.drop(columns=["path_len", "tbv5", "vol5", "up_frac", "lr_mean", "lr2_mean", "ret_skew"])
    if C.REQUIRE_FULL_BARS:
        b = b[b["n_min"] == C.BAR_MINUTES].reset_index(drop=True)
    b["known_at_ts"] = b["bar_open_ts"] + pd.Timedelta(minutes=C.BAR_MINUTES)
    return b


def join_funding(b: pd.DataFrame, sym: str) -> pd.DataFrame:
    try:
        f = load_funding(sym)
    except FileNotFoundError:
        print(f"    no funding file for {sym}; funding features NaN")
        for c in ["last_funding_ts", "last_funding_rate", "last_funding_interval_h", f"fr_mean{C.FUNDING_MEAN_WINDOW}", "fr_std12"]:
            b[c] = np.nan
        b["mins_since_funding"] = np.nan
        return b
    ev = f[["settle_ts", "funding_rate"]].copy()
    ev["interval_h_derived"] = ev["settle_ts"].diff().dt.total_seconds().div(3600)
    ev[f"fr_mean{C.FUNDING_MEAN_WINDOW}"] = ev["funding_rate"].rolling(C.FUNDING_MEAN_WINDOW, min_periods=1).mean()
    ev["fr_std12"] = ev["funding_rate"].rolling(12, min_periods=3).std()
    ev.to_parquet(FUND_DIR / f"{sym}.parquet", index=False)
    ev_j = ev.rename(columns={"settle_ts": "last_funding_ts", "funding_rate": "last_funding_rate",
                              "interval_h_derived": "last_funding_interval_h"})
    b = pd.merge_asof(b.sort_values("known_at_ts"), ev_j, left_on="known_at_ts", right_on="last_funding_ts",
                      direction="backward", allow_exact_matches=False)
    b["mins_since_funding"] = (b["known_at_ts"] - b["last_funding_ts"]).dt.total_seconds() / 60
    assert (b["last_funding_ts"].isna() | (b["last_funding_ts"] < b["known_at_ts"])).all()
    return b


def add_labels(b: pd.DataFrame, slim: pd.DataFrame) -> pd.DataFrame:
    open_map = pd.Series(slim["open"].to_numpy(), index=slim["timestamp"])
    b["entry_ts"] = b["known_at_ts"]
    b["entry_px"] = open_map.reindex(b["entry_ts"]).to_numpy()
    exit_ts = b["known_at_ts"] + pd.Timedelta(minutes=H)
    b[f"exit_ts_{H}"] = exit_ts
    b[f"exit_px_{H}"] = open_map.reindex(exit_ts).to_numpy()
    b[f"y_{H}"] = (b[f"exit_px_{H}"] - b["entry_px"]) / b["atr"]
    return b


def build_symbol(sym: str, do_test: bool) -> bool:
    df = load_ohlcv_1m(sym)
    t0 = df["timestamp"].iloc[0] + pd.Timedelta(hours=C.TRIM_HOURS_AFTER_LISTING)
    df = df[df["timestamp"] >= t0].reset_index(drop=True)
    if len(df) < 60 * 24 * 30:
        print(f"    {sym}: < 30 days after trim, skipped"); return False
    slim = df[["timestamp", "open", "high", "low", "close"]]
    slim.to_parquet(SLIM_DIR / f"{sym}.parquet", index=False)

    b = build_15m(df)
    b = join_funding(b, sym)
    raw = b.copy()
    b = f04.compute_bar_features(b)
    if do_test:
        f04.truncation_test(raw, b, sym)
    b = add_labels(b, slim)
    b["symbol"] = sym
    return_cols = (["symbol", "bar_open_ts", "known_at_ts", "open", "high", "low", "close", "volume", "quote_volume",
                    "atr", "entry_px", f"exit_ts_{H}", f"exit_px_{H}", f"y_{H}"]
                   + f04.BAR_FEATURES + f04.INTRABAR_FEATURES + f04.FUNDING_FEATURES)
    b = b[return_cols]
    fcols = f04.BAR_FEATURES + f04.INTRABAR_FEATURES + f04.FUNDING_FEATURES
    b[fcols] = b[fcols].astype("float32")
    b.to_parquet(FEAT_DIR / f"{sym}.parquet", index=False)
    return True


def main():
    syms = (C.XS_DIR / "build_symbols.txt").read_text(encoding="utf-8").split()
    if C.BTC_SYMBOL not in syms:
        syms = [C.BTC_SYMBOL] + syms
    print(f"{len(syms)} symbols to build")
    t0 = time.time(); tested = 0; built = 0
    for i, s in enumerate(syms):
        if (FEAT_DIR / f"{s}.parquet").exists():
            continue
        try:
            ok = build_symbol(s, do_test=tested < LOOKAHEAD_TEST_SYMBOLS)
            if ok and tested < LOOKAHEAD_TEST_SYMBOLS:
                tested += 1
            built += int(ok)
        except Exception as e:  # noqa
            print(f"    !! {s} failed: {e}")
        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(syms)}  [{time.time() - t0:.0f}s]")

    # cross-asset BTC features appended to every file (same bar_open_ts => same known_at_ts)
    btc = pd.read_parquet(FEAT_DIR / f"{C.BTC_SYMBOL}.parquet", columns=["bar_open_ts", "ret_4_atr", "ret_16_atr", "vol_ratio"])
    btc = btc.rename(columns={"ret_4_atr": "btc_ret_4_atr", "ret_16_atr": "btc_ret_16_atr", "vol_ratio": "btc_vol_ratio"})
    for s in syms:
        fp = FEAT_DIR / f"{s}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        if "btc_ret_4_atr" in d.columns:
            continue
        d = d.merge(btc, on="bar_open_ts", how="left")
        d.to_parquet(fp, index=False)
    print(f"done: {built} built this run, {sum((FEAT_DIR / f'{s}.parquet').exists() for s in syms)} feature files total "
          f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
