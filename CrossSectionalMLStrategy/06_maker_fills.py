"""
Step 19 - Maker-fill simulation for the 4h cross-sectional book.

Same book as step 05 (N per side, inverse-ATR weights, dollar-neutral 2x gross, netting). At each grid
time t the change in each name's weight (dw = w_new - w_prev) is an order:
  reference price  = 1m open at t
  limit            = reference (buy for dw > 0, sell for dw < 0)
  fill             = a 1m bar within the next W minutes trades THROUGH the limit by `through` bps
                     (buy: low <= ref*(1-thr); sell: high >= ref*(1+thr)); fill price = limit; fee = maker rebate
  not filled       = taker at the 1m open at t+W, fee = taker + slippage   (or the change is skipped)
P&L for the period:
  sum_i w_new_i * (exit_i/ref_i - 1)                   position return, reference to reference
  + sum_i dw_i * (ref_i - exec_i)/ref_i                 execution price vs reference (signed; good fills add)
  - sum_i |dw_i| * fee_i                                per-leg fee by execution mode
  - sum_i w_new_i * funding_i                           real settlements in (t, t+4h]
Because the traded portion is priced from the real execution to the real exit, adverse selection on
resting orders is in the number. The un-traded (held) portion is unaffected.

Sweeps W in XS_MAKER_WAITS and through in XS_MAKER_THROUGH; reports fill rate by notional, the blended
cost per side achieved, and net/Sharpe/DD per year, next to the flat taker and maker_mix scenarios.

Outputs: results/xs_maker_report.md, results/xs_maker_equity.png, results/xs_maker_grid.csv
"""
import importlib.util
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config as C

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
_spec = importlib.util.spec_from_file_location("x13", C.PROJECT_DIR / "05_backtest.py")
x13 = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(x13)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TAKER = (C.FEE_TAKER_BPS + C.SLIPPAGE_BPS) / 1e4
MAKER = C.FEE_MAKER_BPS / 1e4
out = []
HS = (f"_h{C.XS_HORIZON}" if C.XS_HORIZON != 240 else "") + ("" if getattr(C, "XS_USE_SYMCAT", True) else "_nosym")


def md(s=""):
    out.append(s)


def np_ts(s):
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_convert("UTC").dt.tz_localize(None)
    return s.to_numpy(dtype="datetime64[ns]")


def load_slims(symbols):
    S = {}
    for sym in symbols:
        fp = C.XS_DIR / "slim" / f"{sym}.parquet"
        if fp.exists():
            d = pd.read_parquet(fp).sort_values("timestamp")
            S[sym] = dict(ts=np_ts(d["timestamp"]), open=d["open"].to_numpy(), high=d["high"].to_numpy(), low=d["low"].to_numpy())
    return S


def fill_orders(slim, t_arr, side_arr, W, thr):
    """vectorised fill check for one symbol. t_arr: decision times; side_arr: +1 buy / -1 sell.
    returns exec_px, mode (0 maker, 1 taker fallback, 2 unfilled/skip), ref_px"""
    ts, op, hi, lo = slim["ts"], slim["open"], slim["high"], slim["low"]
    n = len(ts)
    idx = np.searchsorted(ts, t_arr)
    ok = (idx < n)
    idx_c = np.clip(idx, 0, n - 1)
    ok &= ts[idx_c] == t_arr
    ref = op[idx_c]
    limit = ref.copy()
    filled = np.zeros(len(t_arr), bool)
    for m in range(W):
        j = np.clip(idx + m, 0, n - 1)
        contiguous = ok & (ts[j] == t_arr + np.timedelta64(m, "m")) & ~filled
        hit = np.where(side_arr > 0, lo[j] <= limit * (1 - thr), hi[j] >= limit * (1 + thr))
        filled |= contiguous & hit
    exec_px = np.where(filled, limit, np.nan)
    mode = np.where(filled, 0, 2)
    if C.XS_MAKER_FALLBACK_TAKER:
        j = np.clip(idx + W, 0, n - 1)
        fb = ok & ~filled & (ts[j] == t_arr + np.timedelta64(W, "m"))
        exec_px = np.where(fb, op[j], exec_px)
        mode = np.where(fb, 1, mode)
    exec_px = np.where(ok, exec_px, np.nan)
    return exec_px, mode, np.where(ok, ref, np.nan)


def run_maker(yh, ret, atr, fund, ref_px, slims, n, W, thr):
    Y, R, A, F = yh.to_numpy(), ret.to_numpy(), atr.to_numpy(), fund.to_numpy()
    REF = np.array(ref_px.to_numpy(), dtype=float, copy=True)
    T, S = Y.shape
    syms = list(yh.columns)
    times = np_ts(pd.Series(yh.index))
    # pass 1: target weights and dw per period (identical to step 05 at the same N, k=0)
    Wts = np.zeros((T, S)); w_prev = np.zeros(S)
    for t in range(T):
        w = x13.target_weights(Y[t], A[t], n, 0.0)
        w[np.isnan(R[t])] = 0.0
        Wts[t] = w; w_prev = w
    DW = np.diff(np.vstack([np.zeros(S), Wts]), axis=0)
    # pass 2: execute every non-zero dw per symbol, vectorised
    EXEC = np.full((T, S), np.nan); MODE = np.full((T, S), 2)
    for si, sym in enumerate(syms):
        rows = np.where(np.abs(DW[:, si]) > 1e-12)[0]
        if len(rows) == 0 or sym not in slims:
            continue
        ex, mo, rf = fill_orders(slims[sym], times[rows], np.sign(DW[rows, si]), W, thr)
        EXEC[rows, si] = ex; MODE[rows, si] = mo
        # keep the panel reference consistent with the slim file
        REF[rows, si] = np.where(np.isnan(rf), REF[rows, si], rf)
    # pass 3: P&L
    recs = []
    w_prev = np.zeros(S)
    for t in range(T):
        w = Wts[t].copy(); dw = w - w_prev
        exec_ok = ~np.isnan(EXEC[t]) | (np.abs(dw) < 1e-12)
        # a change that could not be executed at all is not made: keep previous weight for that name
        w = np.where(exec_ok, w, w_prev); dw = w - w_prev
        r = np.nan_to_num(R[t])
        pos_ret = (w * r).sum()
        ref = REF[t]; ex = np.where(np.isnan(EXEC[t]), ref, EXEC[t])
        exec_adj = np.nansum(np.where(np.abs(dw) > 1e-12, dw * (ref - ex) / ref, 0.0))
        fee_leg = np.where(MODE[t] == 0, MAKER, TAKER)
        fees = (np.abs(dw) * fee_leg).sum()
        fnd = (w * np.nan_to_num(F[t])).sum()
        turnover = np.abs(dw).sum()
        maker_notional = np.abs(dw[MODE[t] == 0]).sum()
        recs.append((pos_ret, exec_adj, fees, fnd, pos_ret + exec_adj - fees - fnd, turnover, maker_notional))
        w_prev = w
    D = pd.DataFrame(recs, index=yh.index, columns=["pos_ret", "exec_adj", "fee", "funding", "net", "turnover", "maker_notional"])
    D["gross"] = D["pos_ret"]
    return D


def metrics(D):
    daily = D["net"].groupby(D.index.floor("D")).sum()
    eq = daily.cumsum(); dd = (eq - eq.cummax()).min()
    days = max((D.index.max() - D.index.min()).days, 1)
    tt = D["turnover"].sum()
    return dict(total=float(D.net.sum()), ann=float(D.net.sum() / days * 365),
                sharpe=float(daily.mean() / daily.std() * np.sqrt(365)) if daily.std() > 0 else np.nan, max_dd=float(dd),
                gross=float(D.pos_ret.sum()), exec_adj=float(D.exec_adj.sum()), fees=float(D.fee.sum()), funding=float(D.funding.sum()),
                fill_rate_notional=float(D.maker_notional.sum() / tt) if tt > 0 else np.nan,
                blended_cost_bps_per_side=float((D.fee.sum() - D.exec_adj.sum()) / tt * 1e4) if tt > 0 else np.nan,
                edge_per_turnover_bps=float(D.pos_ret.sum() / tt * 1e4) if tt > 0 else np.nan,
                turnover_per_period=float(D.turnover.mean()))


def main():
    yh, ret, atr, fund = x13.load_panel()
    P = pd.read_parquet(x13.pred_file())
    P = P[P.in_book]
    k = P["known_at_ts"]
    P = P[(k.dt.hour % C.XS_GRID_HOURS == 0) & (k.dt.minute == 0)]
    ref_px = P.pivot(index="known_at_ts", columns="symbol", values="entry_px").reindex(index=yh.index, columns=yh.columns)
    slims = load_slims(yh.columns)
    n = C.XS_MAKER_N
    print(f"panel {len(yh):,} periods, {yh.shape[1]} names, slim files for {len(slims)}")

    md("# Maker-fill simulation for the 4h book\n")
    md(f"N={n}/side, {C.XS_GROSS_LEVERAGE}x gross, grid {C.XS_GRID_HOURS}h, predictions {x13.pred_file().name}. Orders: limit at the 1m open "
       f"at the grid time; fill on trade-through; fallback = {'taker at window end' if C.XS_MAKER_FALLBACK_TAKER else 'skip the change'}. "
       f"Maker {C.FEE_MAKER_BPS} bps, taker {C.FEE_TAKER_BPS}+{C.SLIPPAGE_BPS} bps. Holdout {'INCLUDED' if C.RUN_HOLDOUT else 'excluded'}.\n")

    # flat-cost references from the step-13 engine
    rows, runs = [], {}
    for cname, cb in C.XS_COST_SCENARIOS.items():
        D = x13.run(yh, ret, atr, fund, n, 0.0, cb)
        D = D.rename(columns={"gross": "pos_ret"}); D["exec_adj"] = 0.0; D["maker_notional"] = 0.0; D["gross"] = D["pos_ret"]
        runs[("flat", cname)] = D
        rows.append(dict(execution=f"flat {cname} ({cb} bps/side)", wait_min="-", through_bps="-", **metrics(D)))
    for W in C.XS_MAKER_WAITS:
        for thr in C.XS_MAKER_THROUGH:
            D = run_maker(yh, ret, atr, fund, ref_px.copy(), slims, n, W, thr / 1e4)
            runs[("maker", W, thr)] = D
            m = metrics(D)
            rows.append(dict(execution="maker sim", wait_min=W, through_bps=thr, **m))
            print(f"  W={W:>2} thr={thr}: fill {m['fill_rate_notional']:.0%}  blended cost {m['blended_cost_bps_per_side']:+.2f} bps/side  "
                  f"net {m['total']:+.3f}  sharpe {m['sharpe']:.2f}  dd {m['max_dd']:.3f}")
    G = pd.DataFrame(rows); G.to_csv(C.RESULTS_DIR / f"xs_maker_grid{HS}.csv", index=False)
    md("## Execution scenarios\n"); md(x13.table(G))
    md("\n`blended_cost_bps_per_side` = (fees − execution-price gain) / turnover: the all-in cost per unit traded actually achieved, "
       "including adverse selection on filled limit orders. Compare with `edge_per_turnover_bps`.\n")

    # per-year for the middle maker setting and the flat references
    Wm, thm = C.XS_MAKER_WAITS[len(C.XS_MAKER_WAITS) // 2], C.XS_MAKER_THROUGH[0]
    md(f"\n## Per-year net: flat taker / flat maker_mix / maker sim (W={Wm}, through={thm})\n")
    Dt, Dm, Ds = runs[("flat", "taker")], runs[("flat", "maker_mix")], runs[("maker", Wm, thm)]
    yy = pd.DataFrame({"taker": Dt.net.groupby(Dt.index.year).sum(), "maker_mix": Dm.net.groupby(Dm.index.year).sum(),
                       "maker_sim": Ds.net.groupby(Ds.index.year).sum(),
                       "sim_fill_rate": Ds.maker_notional.groupby(Ds.index.year).sum() / Ds.turnover.groupby(Ds.index.year).sum(),
                       "sim_cost_bps": (Ds.fee.groupby(Ds.index.year).sum() - Ds.exec_adj.groupby(Ds.index.year).sum()) / Ds.turnover.groupby(Ds.index.year).sum() * 1e4,
                       "edge_bps": Ds.pos_ret.groupby(Ds.index.year).sum() / Ds.turnover.groupby(Ds.index.year).sum() * 1e4})
    md(x13.table(yy.reset_index().rename(columns={"known_at_ts": "year", "index": "year"})))

    fig, ax = plt.subplots(figsize=(11, 5), dpi=130)
    fig.patch.set_facecolor("#fcfcfb"); ax.set_facecolor("#fcfcfb")
    ax.plot(Dt.index, Dt.net.cumsum(), lw=1.2, ls="--", color="#52514e", label="flat taker")
    ax.plot(Dm.index, Dm.net.cumsum(), lw=1.2, ls=":", color="#52514e", label="flat maker_mix 1.5 bps")
    i = 0
    for W in C.XS_MAKER_WAITS:
        D = runs[("maker", W, C.XS_MAKER_THROUGH[0])]
        ax.plot(D.index, D.net.cumsum(), lw=1.6, color=PALETTE[i], label=f"maker sim W={W} thr={C.XS_MAKER_THROUGH[0]}"); i += 1
    ax.axhline(0, color="#c3c2b7", lw=0.8); ax.grid(True, color="#e8e8e5", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_title(f"4h book N={n}: simulated maker execution vs flat cost scenarios (cumulative net)", fontsize=11, color="#0b0b0b")
    ax.legend(frameon=False, fontsize=9); fig.tight_layout(); fig.savefig(C.RESULTS_DIR / f"xs_maker_equity{HS}.png"); plt.close(fig)
    md(f"\n![maker equity](xs_maker_equity{HS}.png)\n")
    (C.RESULTS_DIR / f"xs_maker_report{HS}.md").write_text("\n".join(out), encoding="utf-8")
    print(f"written {C.RESULTS_DIR / f'xs_maker_report{HS}.md'}")


if __name__ == "__main__":
    main()
