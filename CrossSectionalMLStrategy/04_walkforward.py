"""
Step 12 - Pooled walk-forward over the training pool; predictions for the book.

Pool construction (once, cached to data_processed/xs/pool.parquet):
  for every symbol in build_symbols.txt, keep only the rows in months where the symbol is in_train.
  Adds the cross-sectional mean of y over the pool at each known_at_ts and y_xs = y - that mean.

Walk-forward (same discipline as step 6):
  test month M: train on pool rows with known_at_ts in [M - TRAIN_MONTHS, M) and exit_ts <= M,
  target = y_xs (XS_DEMEAN_TARGET) or y, clipped to +/- LABEL_CLIP_ATR, Huber, fixed rounds.
  predict every pool row in [M, M+1); rows for in_book symbols are flagged.
  Holdout at/after HOLDOUT_START skipped unless RUN_HOLDOUT.

IC reported cross-sectionally: at each known_at_ts, spearman(yhat, y) across the book names; the
fold IC is the mean of those per-timestamp correlations (this is the number that matters for a
long/short book, not the pooled time-series IC).

Outputs:
  results/xs_predictions.parquet   symbol, known_at_ts, y, y_xs, yhat, in_book, fold
  results/xs_folds.csv             per-fold cross-sectional IC, n_train, n_test
  results/xs_importance.csv
"""
import importlib.util
import sys
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import spearmanr
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_spec = importlib.util.spec_from_file_location("f04", C.PROJECT_DIR / "features_lib.py")
f04 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(f04)
FEATURES = f04.BAR_FEATURES + f04.INTRABAR_FEATURES + f04.FUNDING_FEATURES + f04.CROSS_FEATURES
H = C.XS_HORIZON
YCOL, XCOL = f"y_{H}", f"exit_ts_{H}"
FEAT_DIR = C.XS_DIR / "feat"
TRAIN_M = getattr(C, "XS_TRAIN_MONTHS", C.TRAIN_MONTHS)
SUF = (f"_h{C.XS_HORIZON}" if C.XS_HORIZON != 240 else "") + f"_tr{TRAIN_M}" + ("" if getattr(C, "XS_USE_SYMCAT", True) else "_nosym")


def build_pool() -> pd.DataFrame:
    U = pd.read_parquet(C.XS_DIR / "universe.parquet")
    U["month"] = pd.to_datetime(U["month"], utc=True)
    keep_cols = list(dict.fromkeys(["symbol", "known_at_ts", XCOL, YCOL, "atr", "atr_pct", "entry_px", f"exit_px_{H}"] + FEATURES))
    frames = []
    for sym, g in U[U.in_train].groupby("symbol"):
        fp = FEAT_DIR / f"{sym}.parquet"
        if not fp.exists():
            continue
        d = pd.read_parquet(fp)
        d = d.dropna(subset=[YCOL, "entry_px"])
        d["month"] = d["known_at_ts"].dt.tz_convert("UTC").dt.to_period("M").dt.to_timestamp().dt.tz_localize("UTC")
        gm = g.set_index("month")[["in_book"]]
        d = d.merge(gm, left_on="month", right_index=True, how="inner")   # keeps only in_train months
        frames.append(d[keep_cols + ["in_book"]])
    P = pd.concat(frames, ignore_index=True)
    P = P.dropna(subset=FEATURES)
    xs_mean = P.groupby("known_at_ts")[YCOL].transform("mean")
    P["y_xs"] = P[YCOL] - xs_mean
    P["n_xs"] = P.groupby("known_at_ts")[YCOL].transform("size")
    P = P.sort_values(["known_at_ts", "symbol"]).reset_index(drop=True)
    P.to_parquet(C.XS_DIR / (f"pool_h{H}.parquet" if H != 240 else "pool.parquet"), index=False)
    return P


def xs_ic(df: pd.DataFrame, yhat_col: str, y_col: str) -> float:
    vals = []
    for _, g in df.groupby("known_at_ts"):
        if len(g) >= 5:
            r = spearmanr(g[yhat_col], g[y_col]).correlation
            if not np.isnan(r):
                vals.append(r)
    return float(np.mean(vals)) if vals else np.nan


def main():
    pool_fp = C.XS_DIR / (f"pool_h{H}.parquet" if H != 240 else "pool.parquet")
    if pool_fp.exists():
        P = pd.read_parquet(pool_fp)
        print(f"loaded pool {len(P):,} rows")
    else:
        t0 = time.time(); P = build_pool(); print(f"built pool {len(P):,} rows [{time.time() - t0:.0f}s]")
    P["sym_cat"] = P["symbol"].astype("category")
    use_sym = getattr(C, "XS_USE_SYMCAT", True)
    feats = FEATURES + (["sym_cat"] if use_sym else [])
    cat = ["sym_cat"] if use_sym else "auto"
    print(f"symbol identity feature: {'ON' if use_sym else 'OFF'}")
    target = "y_xs" if C.XS_DEMEAN_TARGET else YCOL

    holdout = pd.Timestamp(C.HOLDOUT_START, tz="UTC")
    last = P["known_at_ts"].max() + pd.Timedelta(minutes=1) if C.RUN_HOLDOUT else holdout
    m0 = pd.Timestamp(C.XS_FIRST_TEST_MONTH, tz="UTC")
    preds, folds, imps = [], [], []
    t0 = time.time()
    while m0 < last:
        m1 = min(m0 + pd.DateOffset(months=C.TEST_MONTHS), last)
        tr = P[(P["known_at_ts"] >= m0 - pd.DateOffset(months=TRAIN_M)) & (P["known_at_ts"] < m0) & (P[XCOL] <= m0)]
        te = P[(P["known_at_ts"] >= m0) & (P["known_at_ts"] < m1)]
        if len(tr) < C.XS_MIN_TRAIN_ROWS or len(te) == 0:
            print(f"  {m0.date()}: skipped (train={len(tr)}, test={len(te)})"); m0 = m1; continue
        model = lgb.LGBMRegressor(**C.LGB_PARAMS)
        model.fit(tr[feats], tr[target].clip(-C.LABEL_CLIP_ATR, C.LABEL_CLIP_ATR), categorical_feature=cat)
        te = te.copy()
        te["yhat"] = model.predict(te[feats])
        ic_book = xs_ic(te[te.in_book], "yhat", YCOL)
        ic_pool = xs_ic(te, "yhat", YCOL)
        pf = te[["symbol", "known_at_ts", YCOL, "y_xs", "yhat", "in_book", "atr", "atr_pct", "entry_px", f"exit_px_{H}", XCOL]].rename(columns={YCOL: "y"})
        pf["fold"] = m0.strftime("%Y-%m")
        preds.append(pf)
        folds.append(dict(fold=m0.strftime("%Y-%m"), n_train=len(tr), n_test=len(te), n_book_rows=int(te.in_book.sum()),
                          ic_xs_book=ic_book, ic_xs_pool=ic_pool, yhat_std=float(te["yhat"].std())))
        imps.append(pd.Series(model.booster_.feature_importance("gain"), index=feats))
        print(f"  {m0.strftime('%Y-%m')} train={len(tr):>8,} test={len(te):>6,} book_rows={int(te.in_book.sum()):>6,} "
              f"IC_xs(book)={ic_book:+.4f} IC_xs(pool)={ic_pool:+.4f} yhat_std={te['yhat'].std():.3f} [{time.time() - t0:.0f}s]")
        m0 = m1

    PR = pd.concat(preds, ignore_index=True)
    PR.to_parquet(C.RESULTS_DIR / f"xs_predictions{SUF}.parquet", index=False)
    F = pd.DataFrame(folds); F.to_csv(C.RESULTS_DIR / f"xs_folds{SUF}.csv", index=False)
    I = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    I.to_csv(C.RESULTS_DIR / f"xs_importance{SUF}.csv", header=["gain"])
    print(f"\n{len(PR):,} predictions over {len(F)} folds. cross-sectional IC on the book: mean {F.ic_xs_book.mean():+.4f}, "
          f"median {F.ic_xs_book.median():+.4f}, folds > 0: {(F.ic_xs_book > 0).mean():.0%}")
    print("top features:\n" + I.head(12).round(0).to_string())


if __name__ == "__main__":
    main()
