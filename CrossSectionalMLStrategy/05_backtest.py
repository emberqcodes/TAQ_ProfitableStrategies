"""
Step 13 - Cross-sectional long/short book backtest on the out-of-sample predictions.

Every XS_GRID_HOURS hours (UTC grid), among the in_book names with a prediction:
  yhat_dm = yhat - mean(yhat over the book)             (removes the market component)
  long  the N names with the highest yhat_dm  (optionally only if yhat_dm >  k)
  short the N names with the lowest  yhat_dm  (optionally only if yhat_dm < -k)
  weights: inverse-ATR% within each leg, each leg scaled to XS_GROSS_LEVERAGE/2 of equity
           (dollar-neutral). Hold to the next grid point (= XS_HORIZON), then re-decide.
  netting: a name kept in the same leg is held; costs are charged on |w_new - w_old| only.
  returns: sum_i w_i * (exit_px_i / entry_px_i - 1) with entry = 1m open at t, exit = 1m open at t+h
  costs:   turnover * cost_bps/1e4 (scenarios in XS_COST_SCENARIOS)
  funding: sum_i w_i * sum(rates settled in (t, t+h])   (longs pay positive rates)

Sensitivity over N in XS_N_LEGS, k in XS_K_LEGS, and cost scenarios. Random-book baseline picks
N random longs and N random shorts from the same names each period, same sizing and costs.

Outputs: results/xs_report.md, results/xs_periods.parquet (main setting), results/xs_equity.png
"""
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
H = C.XS_HORIZON
GRID = f"{C.XS_GRID_HOURS}h"
HS = (f"_h{C.XS_HORIZON}" if C.XS_HORIZON != 240 else "") + ("" if getattr(C, "XS_USE_SYMCAT", True) else "_nosym")
out = []


def pred_file(name="xs_predictions"):
    """prediction file for the configured training window, falling back to the unsuffixed file."""
    suf = f"_tr{getattr(C, 'XS_TRAIN_MONTHS', C.TRAIN_MONTHS)}"
    hs = f"_h{C.XS_HORIZON}" if C.XS_HORIZON != 240 else ""
    suf = suf + ("" if getattr(C, "XS_USE_SYMCAT", True) else "_nosym")
    cands = [C.RESULTS_DIR / f"{name}{hs}{suf}.parquet"] + ([] if hs else [C.RESULTS_DIR / f"{name}.parquet"])
    for cand in cands:
        if cand.exists():
            return cand
    raise FileNotFoundError(f"no {name} file; run 04_walkforward.py")


def md(s=""):
    out.append(s)


def table(df, nd=4):
    d = df.copy()
    for c in d.columns:
        if d[c].dtype.kind == "f":
            d[c] = d[c].round(nd)
    cols = list(d.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in d.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def load_panel():
    P = pd.read_parquet(pred_file())
    P = P[P.in_book].copy()
    k = P["known_at_ts"]
    P = P[(k.dt.hour % C.XS_GRID_HOURS == 0) & (k.dt.minute == 0)]
    P = P.dropna(subset=["entry_px", f"exit_px_{H}", "yhat", "atr_pct"])
    P["ret"] = P[f"exit_px_{H}"] / P["entry_px"] - 1.0
    yh = P.pivot(index="known_at_ts", columns="symbol", values="yhat")
    ret = P.pivot(index="known_at_ts", columns="symbol", values="ret")
    atr = P.pivot(index="known_at_ts", columns="symbol", values="atr_pct")
    # funding settled in (t, t+h], bucketed to the grid period that contains it
    fund = pd.DataFrame(0.0, index=yh.index, columns=yh.columns)
    for sym in yh.columns:
        fp = C.XS_DIR / "fund" / f"{sym}.parquet"
        if not fp.exists():
            continue
        ev = pd.read_parquet(fp)
        per = (ev["settle_ts"] - pd.Timedelta(nanoseconds=1)).dt.floor(GRID)
        s = ev.groupby(per)["funding_rate"].sum()
        s = s.reindex(yh.index).fillna(0.0)
        fund[sym] = s.to_numpy()
    return yh, ret, atr, fund


# ---------------------------------------------------------------------------
# engine
# ---------------------------------------------------------------------------
def target_weights(yh_row, atr_row, n, k, rng=None):
    ok = ~(np.isnan(yh_row) | np.isnan(atr_row))
    idx = np.where(ok)[0]
    w = np.zeros(len(yh_row))
    if len(idx) < C.XS_MIN_NAMES:
        return w
    dm = yh_row[idx] - yh_row[idx].mean()
    if rng is not None:
        perm = rng.permutation(len(idx))
        longs, shorts = idx[perm[:n]], idx[perm[n:2 * n]]
    else:
        order = np.argsort(dm)
        shorts = idx[order[:n]]; longs = idx[order[-n:]]
        if k > 0:
            longs = longs[dm[np.searchsorted(idx, longs)] > k]
            shorts = shorts[dm[np.searchsorted(idx, shorts)] < -k]
    half = C.XS_GROSS_LEVERAGE / 2.0
    for leg, sgn in ((longs, 1.0), (shorts, -1.0)):
        if len(leg) == 0:
            continue
        inv = 1.0 / atr_row[leg]
        w[leg] = sgn * half * inv / inv.sum()
    return w


def run(yh, ret, atr, fund, n, k, cost_bps, seed=None):
    rng = np.random.default_rng(seed) if seed is not None else None
    Y, R, A, F = yh.to_numpy(), ret.to_numpy(), atr.to_numpy(), fund.to_numpy()
    T = len(Y)
    w_prev = np.zeros(Y.shape[1])
    rows = []
    for t in range(T):
        w = target_weights(Y[t], A[t], n, k, rng)
        r = np.nan_to_num(R[t])
        w[np.isnan(R[t])] = 0.0                       # no exit price -> cannot hold this period
        turnover = np.abs(w - w_prev).sum()
        gross = (w * r).sum()
        fee = turnover * cost_bps / 1e4
        fnd = (w * np.nan_to_num(F[t])).sum()
        rows.append((gross, fee, fnd, gross - fee - fnd, turnover, (w[w > 0] * r[w > 0]).sum(), (w[w < 0] * r[w < 0]).sum(),
                     int((w > 0).sum()), int((w < 0).sum())))
        w_prev = w
    D = pd.DataFrame(rows, index=yh.index, columns=["gross", "fee", "funding", "net", "turnover", "long_gross", "short_gross", "n_long", "n_short"])
    return D


def metrics(D):
    daily = D["net"].groupby(D.index.floor("D")).sum()
    eq = daily.cumsum(); dd = (eq - eq.cummax()).min()
    days = max((D.index.max() - D.index.min()).days, 1)
    return dict(periods=len(D), traded_share=float(((D.n_long + D.n_short) > 0).mean()),
                total=float(D.net.sum()), ann=float(D.net.sum() / days * 365),
                sharpe=float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else np.nan,
                max_dd=float(dd), gross_total=float(D.gross.sum()), fees=float(D.fee.sum()), funding=float(D.funding.sum()),
                long_gross=float(D.long_gross.sum()), short_gross=float(D.short_gross.sum()),
                turnover_per_period=float(D.turnover.mean()), hit_rate_period=float((D.net > 0).mean()))


def main():
    yh, ret, atr, fund = load_panel()
    print(f"panel: {len(yh):,} grid periods, {yh.shape[1]} symbols, {yh.index.min()} -> {yh.index.max()}")
    md("# Cross-sectional long/short book\n")
    md(f"Grid {C.XS_GRID_HOURS}h, horizon {H} min, book size {C.XS_BOOK_SIZE}, gross leverage {C.XS_GROSS_LEVERAGE}x "
       f"(dollar-neutral), inverse-ATR weights within legs, netting on rebalance. Cost scenarios (bps per side): "
       f"{C.XS_COST_SCENARIOS}. Holdout {'INCLUDED' if C.RUN_HOLDOUT else 'excluded'}.\n")
    md(f"Panel: {len(yh):,} periods, {yh.shape[1]} distinct book names, {yh.index.min().date()} to {yh.index.max().date()}.\n")

    # ---------------- sensitivity grid
    md("## Sensitivity: names per side (N), threshold k, cost scenario\n")
    rows, runs = [], {}
    for n in C.XS_N_LEGS:
        for k in C.XS_K_LEGS:
            for cname, cb in C.XS_COST_SCENARIOS.items():
                D = run(yh, ret, atr, fund, n, k, cb)
                runs[(n, k, cname)] = D
                rows.append(dict(N=n, k=k, cost=cname, **metrics(D)))
    S = pd.DataFrame(rows)
    md(table(S[["N", "k", "cost", "traded_share", "total", "ann", "sharpe", "max_dd", "fees", "funding", "turnover_per_period", "hit_rate_period"]]))

    # ---------------- main setting
    n_main, k_main = C.XS_N_LEGS[len(C.XS_N_LEGS) // 2], C.XS_K_LEGS[0]
    md(f"\n## Main setting: N={n_main}, k={k_main}, cost={C.XS_REPORT_COST}\n")
    D = runs[(n_main, k_main, C.XS_REPORT_COST)]
    D.to_parquet(C.RESULTS_DIR / f"xs_periods{HS}.parquet")
    m = metrics(D)
    md(table(pd.DataFrame([m])))
    md("\n`long_gross` / `short_gross` are the gross contributions of each leg before costs; if one is persistently "
       "negative the book is not neutral in practice.\n")

    md("\n### Per year (main setting)\n")
    Y = D.groupby(D.index.year).agg(net=("net", "sum"), gross=("gross", "sum"), fees=("fee", "sum"), funding=("funding", "sum"),
                                    long_gross=("long_gross", "sum"), short_gross=("short_gross", "sum"), turnover=("turnover", "mean"))
    daily = D["net"].groupby(D.index.floor("D")).sum()
    Y["sharpe"] = daily.groupby(daily.index.year).apply(lambda x: x.mean() / x.std() * np.sqrt(365) if x.std() > 0 else np.nan)
    md(table(Y.reset_index().rename(columns={"known_at_ts": "year"})))

    md("\n### Per year, all cost scenarios (net)\n")
    yy = pd.DataFrame({c: runs[(n_main, k_main, c)]["net"].groupby(runs[(n_main, k_main, c)].index.year).sum() for c in C.XS_COST_SCENARIOS})
    md(table(yy.reset_index().rename(columns={"known_at_ts": "year", "index": "year"})))

    # ---------------- random-book baseline
    md(f"\n### Random-book baseline ({C.XS_RANDOM_SEEDS} seeds, N={n_main}, same names, sizing and {C.XS_REPORT_COST} costs)\n")
    tots = np.array([run(yh, ret, atr, fund, n_main, 0.0, C.XS_COST_SCENARIOS[C.XS_REPORT_COST], seed=s)["net"].sum()
                     for s in range(C.XS_RANDOM_SEEDS)])
    md(table(pd.DataFrame([dict(strategy=m["total"], random_mean=tots.mean(), random_std=tots.std(),
                                random_p95=np.percentile(tots, 95), strategy_percentile=float((tots < m["total"]).mean()),
                                z=(m["total"] - tots.mean()) / tots.std() if tots.std() > 0 else np.nan)])))

    # ---------------- fragility
    md("\n### Fragility: total after removing the top-N periods (main setting)\n")
    s = np.sort(D["net"].to_numpy())[::-1]
    fr = [dict(removed=0, total=s.sum())] + [dict(removed=n, total=s[n:].sum()) for n in C.FRAGILITY_N if n < len(s)]
    md(table(pd.DataFrame(fr)))

    # ---------------- market context
    md("\n### Context: equal-weight long-only book, 1x, same costs on membership changes\n")
    Ymk = yh.notna().astype(float)
    Wm = Ymk.div(Ymk.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    Rm = ret.fillna(0.0)
    mk_gross = (Wm * Rm).sum(axis=1)
    mk_turn = Wm.diff().abs().sum(axis=1).fillna(Wm.iloc[0].abs().sum())
    mk_net = mk_gross - mk_turn * C.XS_COST_SCENARIOS[C.XS_REPORT_COST] / 1e4
    dm = mk_net.groupby(mk_net.index.floor("D")).sum()
    md(table(pd.DataFrame([dict(total=mk_net.sum(), sharpe=dm.mean() / dm.std() * np.sqrt(365), max_dd=(dm.cumsum() - dm.cumsum().cummax()).min())])))

    # ---------------- IC and features
    F = pd.read_csv(str(pred_file()).replace("xs_predictions", "xs_folds").replace(".parquet", ".csv"))
    md(f"\n### Cross-sectional IC on the book: mean {F.ic_xs_book.mean():+.4f}, median {F.ic_xs_book.median():+.4f}, "
       f"folds > 0: {(F.ic_xs_book > 0).mean():.0%}, worst {F.ic_xs_book.min():+.4f}\n")
    I = pd.read_csv(str(pred_file()).replace("xs_predictions", "xs_importance").replace(".parquet", ".csv"), index_col=0)
    md("### Top 15 features\n")
    md(table(I.head(15).reset_index().rename(columns={"index": "feature"}), 0))

    # ---------------- equity
    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    for i, cname in enumerate(C.XS_COST_SCENARIOS):
        Dd = runs[(n_main, k_main, cname)]
        ax.plot(Dd.index, Dd["net"].cumsum(), lw=1.6, color=PALETTE[i], label=f"{cname} ({C.XS_COST_SCENARIOS[cname]} bps/side)")
    ax.plot(mk_net.index, mk_net.cumsum(), lw=1.2, ls="--", color="#52514e", label="equal-weight long book 1x")
    ax.axhline(0, color="#c3c2b7", lw=0.8); ax.grid(True, color="#e8e8e5", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"Cross-sectional L/S book, N={n_main}/side, {C.XS_GROSS_LEVERAGE}x gross, cumulative net (fraction of equity)", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"xs_equity{HS}.png"); plt.close(fig)
    md("\n![xs equity](xs_equity.png)\n")

    (C.RESULTS_DIR / f"xs_report{HS}.md").write_text("\n".join(out), encoding="utf-8")
    print(f"written {C.RESULTS_DIR / 'xs_report.md'}")
    print(S[["N", "k", "cost", "total", "ann", "sharpe", "max_dd", "turnover_per_period"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
