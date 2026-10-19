"""
Step 4 - Features and labels, with a built-in lookahead self-test.

For each row (one 15m bar, open time T, decision time known_at_ts = T+15m):
  features : computed from 15m bars with bar_open_ts <= T (i.e. known by known_at_ts),
             intra-bar 1m features from inside [T, T+15m), and funding settled strictly
             before known_at_ts (already enforced in step 3).
  labels   : y_<h> = (open_1m[known_at_ts + h] - open_1m[known_at_ts]) / ATR14[T]
             i.e. enter at the first 1m open after the decision, exit at the 1m open h
             minutes later. Both prices are from the slim 1m file.

Lookahead self-test (refuses to write output if it fails):
  1. For random rows, recompute every 15m-derived feature from the data TRUNCATED at that
     row and compare with the full-sample value. Any difference => future data leaked.
  2. Funding: last_funding_ts < known_at_ts for every row.
  3. Labels: entry_ts == known_at_ts, exit_ts == known_at_ts + h, exit_ts > entry_ts.

Output: data_processed/dataset.parquet  (all symbols stacked)
        results/feature_list.json
"""
import json
import numpy as np
import pandas as pd
import config as C

EPS = 1e-12
rng = np.random.default_rng(0)


# ---------------------------------------------------------------------------
# 15m-derived features (all rolling ops are trailing => causal by construction;
# the self-test below verifies it anyway)
# ---------------------------------------------------------------------------
def compute_bar_features(b: pd.DataFrame) -> pd.DataFrame:
    b = b.sort_values("bar_open_ts").reset_index(drop=True).copy()
    c, h, l = b["close"], b["high"], b["low"]
    prev_c = c.shift(1)

    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    b["atr"] = tr.ewm(alpha=1 / C.ATR_PERIOD, adjust=False, min_periods=C.ATR_PERIOD).mean()
    b["atr_pct"] = b["atr"] / c

    for k in C.RET_LAGS:
        b[f"ret_{k}_atr"] = (c - c.shift(k)) / b["atr"]

    lr = np.log(c).diff()
    rv_s = lr.rolling(C.VOL_SHORT).std()
    rv_l = lr.rolling(C.VOL_LONG).std()
    b["vol_ratio"] = rv_s / (rv_l + EPS)
    b["rv_long"] = rv_l

    qv = b["quote_volume"].rolling(C.VWAP_WINDOW).sum()
    vv = b["volume"].rolling(C.VWAP_WINDOW).sum()
    b["vwap_dist_atr"] = (c - qv / (vv + EPS)) / b["atr"]

    delta = c.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / C.RSI_PERIOD, adjust=False, min_periods=C.RSI_PERIOD).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / C.RSI_PERIOD, adjust=False, min_periods=C.RSI_PERIOD).mean()
    b["rsi"] = 100 - 100 / (1 + up / (dn + EPS))

    b["range_atr"] = (h - l) / b["atr"]
    b["body_atr"] = (c - b["open"]) / b["atr"]
    vol_ma = b["volume"].rolling(C.VOL_LONG).mean()
    b["vol_z"] = (b["volume"] - vol_ma) / (b["volume"].rolling(C.VOL_LONG).std() + EPS)
    b["trades_z"] = (b["trades"] - b["trades"].rolling(C.VOL_LONG).mean()) / (b["trades"].rolling(C.VOL_LONG).std() + EPS)

    # calendar (of the bar open; known trivially)
    b["hour"] = b["bar_open_ts"].dt.hour
    b["dow"] = b["bar_open_ts"].dt.dayofweek
    b["min_of_day"] = b["bar_open_ts"].dt.hour * 60 + b["bar_open_ts"].dt.minute
    return b


BAR_FEATURES = (["atr_pct"] + [f"ret_{k}_atr" for k in C.RET_LAGS] +
                ["vol_ratio", "rv_long", "vwap_dist_atr", "rsi", "range_atr", "body_atr",
                 "vol_z", "trades_z", "hour", "dow", "min_of_day"])
INTRABAR_FEATURES = ["ib_taker_buy_ratio", "ib_taker_buy_ratio_last5", "ib_up_frac",
                     "ib_path_len", "ib_net_move", "ib_ret_skew", "ib_vol_last5_share",
                     "ib_ret_std_1m", "ib_chop"]
FUNDING_FEATURES = ["last_funding_rate", f"fr_mean{C.FUNDING_MEAN_WINDOW}", "fr_std12",
                    "mins_since_funding", "last_funding_interval_h"]
CROSS_FEATURES = ["btc_ret_4_atr", "btc_ret_16_atr", "btc_vol_ratio"]


# ---------------------------------------------------------------------------
# labels from the slim 1m file
# ---------------------------------------------------------------------------
def add_labels(b: pd.DataFrame, slim: pd.DataFrame) -> pd.DataFrame:
    open_map = pd.Series(slim["open"].to_numpy(), index=slim["timestamp"])
    b["entry_ts"] = b["known_at_ts"]
    b["entry_px"] = open_map.reindex(b["entry_ts"]).to_numpy()
    for hmin in C.HORIZONS_MIN:
        exit_ts = b["known_at_ts"] + pd.Timedelta(minutes=hmin)
        b[f"exit_ts_{hmin}"] = exit_ts
        b[f"exit_px_{hmin}"] = open_map.reindex(exit_ts).to_numpy()
        b[f"y_{hmin}"] = (b[f"exit_px_{hmin}"] - b["entry_px"]) / b["atr"]
    return b


# ---------------------------------------------------------------------------
# lookahead self-test
# ---------------------------------------------------------------------------
def truncation_test(raw_bars: pd.DataFrame, full_feat: pd.DataFrame, sym: str) -> None:
    n = len(raw_bars)
    idx = rng.choice(np.arange(C.VOL_LONG + 5, n), size=min(C.LOOKAHEAD_TEST_SAMPLES, n - C.VOL_LONG - 5), replace=False)
    worst = 0.0
    for i in idx:
        trunc = compute_bar_features(raw_bars.iloc[: i + 1])
        a = trunc.iloc[-1][BAR_FEATURES].astype(float).to_numpy()
        bfull = full_feat.iloc[i][BAR_FEATURES].astype(float).to_numpy()
        diff = np.nanmax(np.abs(a - bfull))
        worst = max(worst, diff)
        if diff > 1e-8:
            raise AssertionError(f"{sym}: LOOKAHEAD detected at row {i} ({full_feat.iloc[i]['bar_open_ts']}), max diff {diff}")
    print(f"  {sym}: truncation test passed on {len(idx)} rows (max diff {worst:.2e})")


def main() -> None:
    per_sym = {}
    for sym in C.SYMBOLS:
        raw = pd.read_parquet(C.PROCESSED_DIR / f"{sym}_15m_f.parquet")
        feat = compute_bar_features(raw)
        truncation_test(raw, feat, sym)

        assert (feat["last_funding_ts"].isna() | (feat["last_funding_ts"] < feat["known_at_ts"])).all(), f"{sym}: funding lookahead"

        slim = pd.read_parquet(C.PROCESSED_DIR / f"{sym}_1m_slim.parquet")
        feat = add_labels(feat, slim)
        for hmin in C.HORIZONS_MIN:
            assert (feat[f"exit_ts_{hmin}"] > feat["entry_ts"]).all()
            assert ((feat[f"exit_ts_{hmin}"] - feat["entry_ts"]) == pd.Timedelta(minutes=hmin)).all()
        assert (feat["entry_ts"] == feat["known_at_ts"]).all()
        per_sym[sym] = feat

    # cross-asset: BTC's own features aligned on the same bar_open_ts (same known_at_ts)
    btc = per_sym[C.BTC_SYMBOL][["bar_open_ts", "ret_4_atr", "ret_16_atr", "vol_ratio"]].rename(columns={
        "ret_4_atr": "btc_ret_4_atr", "ret_16_atr": "btc_ret_16_atr", "vol_ratio": "btc_vol_ratio"})
    frames = []
    for sym, f in per_sym.items():
        f = f.merge(btc, on="bar_open_ts", how="left")
        frames.append(f)
    ds = pd.concat(frames, ignore_index=True)

    all_feats = BAR_FEATURES + INTRABAR_FEATURES + FUNDING_FEATURES + CROSS_FEATURES
    label_cols = [f"y_{h}" for h in C.HORIZONS_MIN]

    n0 = len(ds)
    ds = ds.dropna(subset=all_feats + ["entry_px", "atr"])
    ds = ds.dropna(subset=label_cols, how="all")   # keep rows with at least one label
    ds = ds.sort_values(["known_at_ts", "symbol"]).reset_index(drop=True)
    print(f"dataset rows: {len(ds):,} (dropped {n0 - len(ds):,} warm-up / edge rows)")

    keep = (["symbol", "bar_open_ts", "known_at_ts", "open", "high", "low", "close", "volume",
             "atr", "entry_ts", "entry_px"] + all_feats +
            [c for h in C.HORIZONS_MIN for c in (f"exit_ts_{h}", f"exit_px_{h}", f"y_{h}")])
    ds[keep].to_parquet(C.PROCESSED_DIR / "dataset.parquet", index=False)

    (C.RESULTS_DIR / "feature_list.json").write_text(json.dumps({
        "features": all_feats, "labels": label_cols,
        "n_rows": int(len(ds)),
        "per_symbol": ds["symbol"].value_counts().to_dict(),
        "span": [str(ds["known_at_ts"].min()), str(ds["known_at_ts"].max())],
    }, indent=2))

    # quick sanity summary for you to send back
    print("\nlabel summary (ATR units):")
    print(ds.groupby("symbol")[label_cols].describe().T.round(3).to_string())
    print("\nfeature NaN check:", ds[all_feats].isna().sum().sum())
    print(f"\nwritten {C.PROCESSED_DIR / 'dataset.parquet'} and {C.RESULTS_DIR / 'feature_list.json'}")


if __name__ == "__main__":
    main()
